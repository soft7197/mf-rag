import os
import json
import subprocess
import javalang
import re
from sympy import true


from lib import prompt_maker, predictor, extracting_patches_from_output, query_vector_db, evaluation


# Configuration
json_path = ""  # path to the JSON file with bug data
output_dir = "s"
checkout_dir = "s"                     # base directory to store outputs per project                       # number of similar methods to include when needed
MAX_ITERATIONS = 10                        # maximum iterations for trying patches
BEAM_SIZE = 10                       
# Load bug data from JSON
with open(json_path, 'r') as f:
    bugs_data = json.load(f)
to_process = {}
for bug in bugs_data:
    if  os.path.exists(f"{output_dir}/{bug.split('-')[0]}/{bug}.json"):
            with open(f"{output_dir}/{bug.split('-')[0]}/{bug}.json", 'r') as file:
                patch_data = json.load(file)
            for method in patch_data[bug]['functions']:
                if 'generation' in method:
                    # if '3' in method['generation']:
                    #     print(bug)
                    if '2' in method['generation']:
                        if 'plausible_patches' not in method['generation']['2']:
                            if 'directly_related_tests' in method and len(method['directly_related_tests']) > 1:
                                to_process[bug] = patch_data[bug]
                                break
                            elif 'non_directly_related_tests' in method and len(method['non_directly_related_tests'])>1:
                                to_process[bug] = patch_data[bug]
                                break
                    elif '1' in method['generation']:
                        if 'plausible_patches' not in method['generation']['1']:
                            if 'directly_related_tests' in method and len(method['directly_related_tests']) > 1:
                                to_process[bug] = patch_data[bug]
                                break
                            elif 'non_directly_related_tests' in method and len(method['non_directly_related_tests'])>1:
                                to_process[bug] = patch_data[bug]
                                break
                
            # plausible=0
            # norel = 0
            # for method in patch_data[bug]['functions']:
            #     if 'generation' in method and '1' in method['generation']:
            #         if 'plausible_patches' in method['generation']['1']:
            #             plausible+=1
            #     if 'no_relatable_failing_tests' in method:
            #         norel+=1
            # if plausible + norel != patch_data[bug]['function_num']:
            #     to_process[bug] = patch_data[bug]
                    
                                   
os.makedirs(output_dir, exist_ok=True)
# Set OpenAI API key from environment (ensure OPENAI_API_KEY is defined)

def apply_patch_to_code(project_dir, bug_info, patch_text):
    """Apply the GPT-generated patch text to the source files in the project directory."""
    # Remove markdown fences (```java) from the patch text if present
    patch_lines = [line for line in patch_text.splitlines() if not line.strip().startswith("```")]
    clean_patch = "\n".join(patch_lines)
    # Split patch text into separate method code blocks by matching braces
    method_blocks = []
    braces = 0
    current_block = ""
    for char in clean_patch:
        if char == '{':
            braces += 1
        if char == '}':
            braces -= 1
        current_block += char
        # A method block ends when braces return to zero after starting
        if braces == 0 and current_block.strip():
            method_blocks.append(current_block.strip())
            current_block = ""
    if not method_blocks:
        method_blocks = [clean_patch.strip()]  # fallback: treat entire patch as one block
    # Pad or truncate method_blocks to match the number of buggy functions
    num_funcs = len(bug_info.get("functions", []))
    if len(method_blocks) < num_funcs:
        # If fewer blocks than functions, assume missing ones were not provided (leave them unchanged)
        method_blocks += [""] * (num_funcs - len(method_blocks))
    else:
        method_blocks = method_blocks[:num_funcs]
    # Apply each patch block to the corresponding file and location
    for block, func in zip(method_blocks, bug_info.get("functions", [])):
        if block == "":
            continue  # no patch for this function (skip)
        file_path = os.path.join(project_dir, func["path"])
        start_line = func["start_loc"]
        end_line   = func["end_loc"]
        # Replace the lines [start_line, end_line] in the file with the patch block
        try:
            with open(file_path, 'r') as f:
                file_lines = f.readlines()
            # Convert to 0-based indices for slicing
            s_idx = start_line - 1
            e_idx = end_line - 1
            # Ensure the patch block ends with a newline
            if not block.endswith("\n"):
                block += "\n"
            # Replace the specified lines with the new code block
            new_file_lines = file_lines[:s_idx] + [block] + file_lines[e_idx+1:]
            with open(file_path, 'w') as f:
                f.writelines(new_file_lines)
        except Exception as e:
            print(f"Error applying patch to {file_path}: {e}")

def save_iteration_logs(bug_id, iteration, prompt, patch, test_output, status):
    """Save prompt, patch, test output, and status of an iteration to files in the bug's output directory."""
    bug_output_dir = os.path.join(output_dir, bug_id)
    os.makedirs(bug_output_dir, exist_ok=True)
    with open(os.path.join(bug_output_dir, f"iter{iteration}_prompt.txt"), 'w') as f:
        f.write(prompt)
    with open(os.path.join(bug_output_dir, f"iter{iteration}_patch.java"), 'w') as f:
        f.write(patch)
    with open(os.path.join(bug_output_dir, f"iter{iteration}_test_output.txt"), 'w') as f:
        f.write(test_output)
    with open(os.path.join(bug_output_dir, f"iter{iteration}_status.txt"), 'w') as f:
        f.write(status)

import concurrent.futures
import multiprocessing

def process_bug(bug_id_bug_info_tuple):
    import os
    import json
    import shutil

    # Re-import other required modules here if needed
    bug_id, bug_info = bug_id_bug_info_tuple

    project = bug_id.split('-')[0]
    bug_num = bug_id.split('-')[1] if '-' in bug_id else ""
    print(f"\n=== Processing {bug_id} ===")
    work_dir = os.path.join(checkout_dir, project, bug_id)
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    if bug_num:
        try:
            evaluation.checkout_project_buggy(project, bug_num, work_dir)
        except Exception as e:
            return
    
    for iteration in range(2, 3):
        include_similar = (iteration > 1)
        save_dir = os.path.join(output_dir, f"{bug_id.split('-')[0]}")
        save_file = os.path.join(save_dir, f"{bug_id}.json")
        
        if os.path.exists(save_file):
            with open(save_file, 'r') as f:
                patches_saved = json.load(f)
        else:
            patches_saved = {}

        direct_methods = [item for item in patches_saved[bug_id]['functions'] if 'directly_related_tests' in item]
        non_direct_methods = [item for item in patches_saved[bug_id]['functions'] if 'non_directly_related_tests' in item and 'no_relatable_failing_tests' not in item]
        methods_without_testcase = [item for item in patches_saved[bug_id]['functions'] if 'no_relatable_failing_tests' in item] 
        
        #generation for direct methods 

        for method in direct_methods:
            if 'generation' in method and str(iteration-1) in method['generation']:
                if 'patches' not in method['generation'][str(iteration-1)]:
                    continue
                if 'plausible_patches' in method['generation'][str(iteration-1)]:
                    continue
            else:
                continue
            if include_similar:
                retrieved_context = query_vector_db.retrieve_context(bug_id, [method])
            else:
                retrieved_context = []
            failing_tests = [bug_info['trigger_test'][test] for test in bug_info['trigger_test'] if test in method['directly_related_tests']]
            prompt = prompt_maker.build_prompt_sm(bug_id, [method['comment'], method['buggy_function']], failing_tests, retrieved_context , include_similar)
            if 'generation' not in method:
                method['generation'] = {}
                method['generation'][str(iteration)]= {} 
                method['generation'][str(iteration)]['prompt'] = prompt 
            else:
                method['generation'][str(iteration)] = {}
                method['generation'][str(iteration)]['prompt'] = prompt

            print(f"Iteration {iteration}: Requesting patch from GPT-4{' (with similar methods)' if include_similar else ''}...")
            try:
                responses = predictor.get_response(prompt, BEAM_SIZE)
            except Exception as api_error:
                print(f"OpenAI API call failed at iteration {iteration} for {bug_id}: {api_error}")
                break
            extracted_patches = []
            for patch in responses:
                extracted_patches.append(extracting_patches_from_output.extract_fixed_method(patch)[0])

            method['generation'][str(iteration)]['patches'] = extracted_patches
        

        #generation for non direct methods 

        methods = [method for method in non_direct_methods if 'generation' in method and str(iteration-1) in method['generation'] and 'plausible_patches' not in method['generation'][str(iteration-1)]]
        methods_for_prompt = [[method['comment'], method['buggy_function']] for method in methods]
        if len(methods) > 0:
            failing_testcases = []
            for method in non_direct_methods:
                for test in method['non_directly_related_tests']:
                    failing_testcases.append(test)
            if include_similar:
                retrieved_context = query_vector_db.retrieve_context(bug_id, methods)
            else:
                retrieved_context = []
            failing_tests = [bug_info['trigger_test'][test] for test in bug_info['trigger_test'] if test in failing_testcases]     
            if len(failing_tests) != 0: 
                prompt = prompt_maker.build_prompt_mm(bug_id, methods_for_prompt, failing_tests, retrieved_context , include_similar)
                for method in non_direct_methods:
                    if 'generation' not in method:
                        method['generation'] = {}
                        method['generation'][str(iteration)]= {} 
                        method['generation'][str(iteration)]['prompt'] = prompt 
                    else:
                        method['generation'][str(iteration)]= {}
                        method['generation'][str(iteration)]['prompt'] = prompt
                print(f"Iteration {iteration}: Requesting patch from GPT-4{' (with similar methods)' if include_similar else ''}...")
                try:
                    responses = predictor.get_response(prompt, BEAM_SIZE)
                except Exception as api_error:
                    print(f"OpenAI API call failed at iteration {iteration} for {bug_id}: {api_error}")
                    break
                for patch in responses:
                    extracted_patches = extracting_patches_from_output.extract_fixed_method(patch)
                    index = 0
                    for method in non_direct_methods:
                        if 'patches' not in method['generation'][str(iteration)]:
                            method['generation'][str(iteration)]['patches'] = []
                        try:    
                            method['generation'][str(iteration)]['patches'].append(extracted_patches[index])
                        except:
                            print("no extracted patches")
                        index+=1
            else:
                failing_tests = [bug_info['trigger_test'][test] for test in bug_info['trigger_test']]
                prompt = prompt_maker.build_prompt_mm(bug_id, methods_for_prompt, failing_tests, retrieved_context , include_similar)
                for method in non_direct_methods:
                    if 'generation' not in method:
                        method['generation'] = {}
                        method['generation'][str(iteration)]= {} 
                        method['generation'][str(iteration)]['prompt'] = prompt 
                    else:
                        method['generation'][str(iteration)]= {}
                        method['generation'][str(iteration)]['prompt'] = prompt
                print(f"Iteration {iteration}: Requesting patch from GPT-4{' (with similar methods)' if include_similar else ''}...")
                try:
                    responses = predictor.get_response(prompt, BEAM_SIZE)
                except Exception as api_error:
                    print(f"OpenAI API call failed at iteration {iteration} for {bug_id}: {api_error}")
                    break
                for patch in responses:
                    extracted_patches = extracting_patches_from_output.extract_fixed_method(patch)
                    index = 0
                    for method in non_direct_methods:
                        if 'patches' not in method['generation'][str(iteration)]:
                            method['generation'][str(iteration)]['patches'] = []
                        try:    
                            method['generation'][str(iteration)]['patches'].append(extracted_patches[index])
                        except:
                            print("no extracted patches")
                        index+=1
        patches_saved[bug_id]['context_generation_done'] = True 
        os.makedirs(save_dir, exist_ok=True)
        with open(save_file, 'w') as f:
            json.dump(patches_saved, f, indent=4)

def extract_failed_test_cases_with_optional_errors(log):
    failure_pattern = re.compile(r"--- ([\w\.]+)::([\w\d_]+)")
    error_pattern = re.compile(r"(Exception|Error):.*")

    lines = log.strip().splitlines()
    results = []
    current_test = None
    found_error = False

    for line in lines:
        line = line.strip()
        match = failure_pattern.match(line)
        if match:
            if current_test and not found_error:
                results.append({
                    "failing_test": current_test,
                    "error_message": None
                })
            suite, test = match.groups()
            current_test = f"{suite}::{test}"
            found_error = False
            continue

        if current_test and error_pattern.search(line):
            results.append({
                "failing_test": current_test,
                "error_message": line
            })
            current_test = None
            found_error = True

    if current_test and not found_error:
        results.append({
            "failing_test": current_test,
            "error_message": None
        })

    return results

def tokenize_java_method(method_code):
    try:
        tokens = list(javalang.tokenizer.tokenize(method_code))
        return [token.value for token in tokens]
    except:
        return []  # skip invalid Java

def remove_duplicate_methods(method_list):
    seen = set()
    unique_methods = []

    for method in method_list:
        tokens = tokenize_java_method(method)
        token_str = ' '.join(tokens)

        if token_str not in seen:
            seen.add(token_str)
            unique_methods.append(method)

    return unique_methods

def evaluate(bug_id_bug_info_tuple):
     
    iteration  = '2'

    bug_id, bug_info = bug_id_bug_info_tuple
    
    if not os.path.exists(f"{output_dir}/{bug_id.split('-')[0]}/{bug_id}.json"):
        return
    
    checkout_dir = "/home/selab/Desktop/MF-dlt/checkouts"
    evaluation.checkout_project_buggy(bug_id.split('-')[0], bug_id.split('-')[1], f"{checkout_dir}/{bug_id}")
    evaluation.run_tests(f"{checkout_dir}/{bug_id}")
    if not os.path.exists(f"{checkout_dir}/{bug_id}/failing_tests"):
        return    
    with open(f"{checkout_dir}/{bug_id}/failing_tests", 'r') as file:
        log1 = file.read()
    all_failing_tests = evaluation.extract_failed_test_cases(log1)                    

    with open(f"{output_dir}/{bug_id.split('-')[0]}/{bug_id}.json","r") as file:
        patch_data = json.load(file)
    
    direct_methods = [item for item in patch_data[bug_id]['functions'] if 'directly_related_tests' in item]
    non_direct_methods = [item for item in patch_data[bug_id]['functions'] if 'non_directly_related_tests' in item and 'no_relatable_failing_tests' not in item]
    methods_without_testcase = [item for item in patch_data[bug_id]['functions'] if 'no_relatable_failing_tests' in item] 
    
    
    print(f"{bug_id} evaluation started")
    direct_tests = []
    for method in direct_methods:
        for test in method['directly_related_tests']:
            direct_tests.append(test)
        patches = method['generation'][iteration]['patches']
        patches = remove_duplicate_methods(patches)
        for patch in patches:
            evaluation.apply_patch(f"{checkout_dir}/{bug_id}/{method['path']}", method['buggy_function'], patch, method['start_loc'], method['end_loc'])
            evaluation.run_tests(f"{checkout_dir}/{bug_id}")
            evaluation.apply_patch(f"{checkout_dir}/{bug_id}/{method['path']}", patch, method['buggy_function'])
        
            if not os.path.exists(f"{checkout_dir}/{bug_id}/failing_tests"):
                continue

            with open(f"{checkout_dir}/{bug_id}/failing_tests", 'r') as file:
                log = file.read()
            failing_tests = evaluation.extract_failed_test_cases(log)

            if set(method['directly_related_tests']) & set(failing_tests): 
                continue

            if 'plausible_patches' not in method['generation'][iteration]:
                method['generation'][iteration]['plausible_patches'] = []
            method['generation'][iteration]['plausible_patches'].append(patch)
        method['generation'][iteration]['evaluation_done'] = True
    try:
        if len(non_direct_methods) != 0:
            for i in range(10):
                for method in non_direct_methods:
                    if 'patches' not in method['generation'][iteration]:
                        continue
                    patches = method['generation']['1']['patches']
                    if len(patches)>=i+1:
                        evaluation.apply_patch(f"{checkout_dir}/{bug_id}/{method['path']}", method['buggy_function'], patches[i] , method['start_loc'], method['end_loc'])
                evaluation.run_tests(f"{checkout_dir}/{bug_id}")
                
                if not os.path.exists(f"{checkout_dir}/{bug_id}/failing_tests"):
                    continue

                with open(f"{checkout_dir}/{bug_id}/failing_tests", 'r') as file:
                    log = file.read()
                failing_tests = evaluation.extract_failed_test_cases(log)
                

                if set(failing_tests) != set(direct_tests):
                    continue

                for method in non_direct_methods:
                    patches = method['generation'][iteration]['patches']
                    if len(patches)>=i+1:
                        if 'plausible_patches' not in method['generation'][iteration]:
                            method['generation'][iteration]['plausible_patches'] = []
                        method['generation'][iteration]['plausible_patches'].append(patches[i])
            method['generation'][iteration]['evaluation_done'] = True
    except:
        print("Error in non_related_methods")
    patch_data[bug_id]['evaluation_done'] = True
    patch_data[bug_id]['evaluation_done2'] = True
    with open(f"{output_dir}/{bug_id.split('-')[0]}/{bug_id}.json","w") as file:
        json.dump(patch_data, file, indent=4)
    print(f"{bug_id} evaluation ended")

def parse_defects4j_test_output(output: str) -> dict:

    result = {
        "status": "",
        "failing_tests": [],
        "raw_output": output
    }

    if "Cannot compile test suite!" in output or "BUILD FAILED" in output:
        result["status"] = "compilation_error"
        return result

    # Check for no failing tests FIRST
    if re.search(r"Failing tests:\s*0", output):
        result["status"] = "no_failures"
        return result

    # Then check for actual failing test cases
    if "Failing tests:" in output:
        test_matches = re.findall(r"- ([\w\.\$]+::[\w\d_]+)", output)
        if test_matches:
            result["status"] = "failing_tests"
            result["failing_tests"] = test_matches
        else:
            result["status"] = "no_failures"
        return result

    result["status"] = "unknown"
    return result

import os
import json
import shutil

def process_bug_feedback(bug_id_bug_info_tuple):

    # Re-import other required modules here if needed
    bug_id, bug_info = bug_id_bug_info_tuple

    project = bug_id.split('-')[0]
    bug_num = bug_id.split('-')[1] if '-' in bug_id else ""
    #print(f"\n=== Processing {bug_id} ===")
    work_dir = os.path.join(checkout_dir, bug_id)
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # if bug_num:
    #     try:
    #         evaluation.checkout_project_buggy(project, bug_num, work_dir)
    #     except Exception as e:
    #         return
    
    for iteration in range(3, 4):
        include_similar = (iteration > 1)
        save_dir = os.path.join(output_dir, f"{bug_id.split('-')[0]}")
        save_file = os.path.join(save_dir, f"{bug_id}.json")
        
        if os.path.exists(save_file):
            with open(save_file, 'r') as f:
                patches_saved = json.load(f)
        else:
            patches_saved = {}

        direct_methods = [item for item in patches_saved[bug_id]['functions'] if 'directly_related_tests' in item]
        non_direct_methods = [item for item in patches_saved[bug_id]['functions'] if 'non_directly_related_tests' in item and 'no_relatable_failing_tests' not in item]
        methods_without_testcase = [item for item in patches_saved[bug_id]['functions'] if 'no_relatable_failing_tests' in item] 
        if len(direct_methods) == 0 and len(methods_without_testcase)==0 :
            print(bug_id)

        for method in direct_methods:
            if 'generation' not in method:
                continue
            if '2' not in method['generation']:
                continue
            if 'plausible_patches' in method['generation']['2']:
                continue
            patches = method['generation']['2']['patches']
            new_failing_tests = []
            failing_test_minimum = len(method['directly_related_tests'])
            comp =0
            for patch in patches:
                evaluation.apply_patch(f"{checkout_dir}/{bug_id}/{method['path']}", method['buggy_function'], patch, method['start_loc'], method['end_loc'])
                output = evaluation.run_tests(f"{checkout_dir}/{bug_id}")
                evaluation.apply_patch(f"{checkout_dir}/{bug_id}/{method['path']}", patch, method['buggy_function'])
                
                test_result = parse_defects4j_test_output(output)

                if test_result['status'] == 'compilation_error' or 'Timeout' in test_result['raw_output']:
                    comp += 1
                    continue
                elif test_result['status'] == 'failing_tests':
                    raw_failing_tests = test_result['failing_tests']
                    failing_tests_path = f"{checkout_dir}/{bug_id}/failing_tests"
                    if os.path.exists(failing_tests_path):                   
                        with open(failing_tests_path, 'r') as file:
                            log = file.read()
                        file_failing_tests = extract_failed_test_cases_with_optional_errors(log)
                    else:
                        file_failing_tests = []
                    if len(file_failing_tests) != len(raw_failing_tests):
                        current_failing_tests_errors = []
                        for test in raw_failing_tests:
                            current_failing_tests_errors.append({'failing_test': test, 'error_message': None})
                    else:
                        current_failing_tests_errors = file_failing_tests
                elif test_result['status'] == 'no_failures':
                    patch_data[bug_id][str(iteration-1)].setdefault('plausible_patches', []).append(patch)
                    failing_test_minimum = 0
                    continue                                     
                 
                for test in current_failing_tests_errors:
                    if test['failing_test'] not in bug_info['trigger_test']:
                        existing = [test['failing_test'] for test in new_failing_tests]
                        if test['failing_test'] not in existing:
                            new_failing_tests.append({'failing_test': test['failing_test'], 'error_message' : test['error_message']})
                       
                if len(current_failing_tests_errors) < failing_test_minimum:
                    if str(iteration) not in patch_data[bug_id]:
                        patch_data[bug_id][str(iteration)] = {}
                    patch_data[bug_id][str(iteration)]['best_patch_for_prompt'] = patch
                    patch_data[bug_id][str(iteration)]['remaining_failing_tests'] = [test['failing_test'] for test in current_failing_tests_errors]
                    failing_test_minimum = len(current_failing_tests_errors)
        
        os.makedirs(save_dir, exist_ok=True)
        with open(save_file, 'w') as f:
            json.dump(patches_saved, f, indent=4)

def evaluate_rest(bug_id_bug_info_tuple):

    bug_id, bug_info = bug_id_bug_info_tuple
    
    if not os.path.exists(f"{output_dir}/{bug_id.split('-')[0]}/{bug_id}.json"):
        return
    
    checkout_dir = "/home/selab/Desktop/MF-dlt/checkouts"
    evaluation.checkout_project_buggy(bug_id.split('-')[0], bug_id.split('-')[1], f"{checkout_dir}/{bug_id}")
    with open(f"{output_dir}/{bug_id.split('-')[0]}/{bug_id}.json","r") as file:
        patch_data = json.load(file)
    
    direct_methods = [item for item in patch_data[bug_id]['functions'] if 'directly_related_tests' in item]
    non_direct_methods = [item for item in patch_data[bug_id]['functions'] if 'non_directly_related_tests' in item and 'no_relatable_failing_tests' not in item]
    methods_without_testcase = [item for item in patch_data[bug_id]['functions'] if 'no_relatable_failing_tests' in item] 
    for method in patch_data[bug_id]['functions']:
            if 'generation' in method and '1' in method['generation']:
                if 'plausible_patches' in method['generation']['1']:
                    patch = method['generation']['1']['plausible_patches'][0]
                    evaluation.apply_patch(f"{checkout_dir}/{bug_id}/{method['path']}", method['buggy_function'], patch, method['start_loc'], method['end_loc'])
                else:
                    candidate_method = method   
    
    for iteration in range(1,3):    
        if str(iteration) not in candidate_method['generation']:
            continue 
        if 'patches' not in candidate_method['generation'][str(iteration)]:
            print(f"{bug_id}---no patches in some method----------------------------------------------------")
            continue
        patches = candidate_method['generation'][str(iteration)]['patches']
        patches = remove_duplicate_methods(patches)
        for patch in patches:
            evaluation.apply_patch(f"{checkout_dir}/{bug_id}/{candidate_method['path']}", candidate_method['buggy_function'], patch, candidate_method['start_loc'], candidate_method['end_loc'])
            evaluation.run_tests(f"{checkout_dir}/{bug_id}")
            evaluation.apply_patch(f"{checkout_dir}/{bug_id}/{candidate_method['path']}", patch, candidate_method['buggy_function'])
        
            if not os.path.exists(f"{checkout_dir}/{bug_id}/failing_tests"):
                continue

            with open(f"{checkout_dir}/{bug_id}/failing_tests", 'r') as file:
                log = file.read()
            failing_tests = evaluation.extract_failed_test_cases(log)

            if len(failing_tests) != 0: 
                continue

            if 'plausible_patches' not in candidate_method['generation'][str(iteration)]:
                candidate_method['generation'][str(iteration)]['plausible_patches'] = []
            candidate_method['generation'][str(iteration)]['plausible_patches'].append(patch)

    patch_data[bug_id]['evaluation_done'] = True
    patch_data[bug_id]['evaluation_done2'] = True
    with open(f"{output_dir}/{bug_id.split('-')[0]}/{bug_id}.json","w") as file:
        json.dump(patch_data, file, indent=4)
    print(f"{bug_id} evaluation ended")
# Outside the function
if __name__ == "__main__":

    for bug in to_process.items():
        b,i=bug
        process_bug_feedback(bug)
        # if (b not in plausible_bugs) and (b not in plausible_no_relatable):
        #     to_process[b] =  bug
        #     print(b)
        #process_bug_feedback(bug)
    
    # max_workers = multiprocessing.cpu_count()  # Use all available cores
    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     executor.map(process_bug_feedback, to_process.items())

    # max_workers = multiprocessing.cpu_count()  # Use all available cores
    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     executor.map(evaluate, to_process.items())
