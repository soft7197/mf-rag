import os
import json
import subprocess

from sympy import true


from lib import prompt_maker, predictor, extracting_patches_from_output, query_vector_db, evaluation


# Configuration
json_path = ""  # path to the JSON file with bug data
output_dir = ""                     # base directory to store outputs per project                       # number of similar methods to include when needed
MAX_ITERATIONS = 1                        # maximum iterations for trying patches
BEAM_SIZE = 10                       

# Load bug data from JSON
with open(json_path, 'r') as f:
    bugs_data = json.load(f)
to_process = {}
for bug in bugs_data:
    if  os.path.exists(f"{output_dir}/{bug.split('-')[0]}/{bug}.json"):
            with open(f"{output_dir}/{bug.split('-')[0]}/{bug}.json", 'r') as file:
                patch_data = json.load(file)
            if 'functions' in patch_data[bug]:
                for method in patch_data[bug]['functions']:
                    if 'no_relatable_failing_tests' in  method:
                        if 'generation_done_for_notest_methods' not in patch_data[bug]:
                            to_process[bug] = patch_data[bug]
                        continue

os.makedirs(output_dir, exist_ok=True)

def run_defects4j_command(command, cwd):
    """Run a defects4j command in the given working directory and return (exit_code, output)."""
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as e:
        return 1, f"Command timed out: {e}"
    output = result.stdout + result.stderr
    return result.returncode, output

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
    work_dir = os.path.join(output_dir, "checkouts", bug_id)
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    if bug_num:
        cmd = ["defects4j", "checkout", "-p", project, "-v", f"{bug_num}b", "-w", work_dir]
        ret, cout = run_defects4j_command(cmd, cwd=".")
        if ret != 0:
            print(f"Error: Failed to checkout {bug_id} - {cout.strip()}")
            return
        

    direct_methods = [item for item in bug_info['functions'] if 'directly_related_tests' in item]
    non_direct_methods = [item for item in bug_info['functions'] if 'non_directly_related_tests' in item and 'no_relatable_failing_tests' not in item]
    methods_without_testcase = [item for item in bug_info['functions'] if 'no_relatable_failing_tests' in item] 
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        include_similar = (iteration > 1)
        save_dir = os.path.join(output_dir, f"{bug_id.split('-')[0]}")
        save_file = os.path.join(save_dir, f"{bug_id}.json")
        
        if os.path.exists(save_file):
            with open(save_file, 'r') as f:
                patches_saved = json.load(f)
        else:
            patches_saved = {}
        if include_similar:
            retrieved_context = query_vector_db.retrieve_context(bug_id)
        else:
            retrieved_context = []
        
        #generation for direct methods 

        for method in direct_methods:
            failing_tests = [bug_info['trigger_test'][test] for test in bug_info['trigger_test'] if test in method['directly_related_tests']]
            prompt = prompt_maker.build_prompt_sm(bug_id, [method['comment'], method['buggy_function']], failing_tests, retrieved_context , include_similar)
            if 'generation' not in method:
                method['generation'] = {}
                method['generation'][iteration] = {} 
                method['generation'][iteration]['prompt'] = prompt 
            else:
                method['generation']['iteration'] = {}
                method['generation'][iteration]['prompt'] = prompt

            print(f"Iteration {iteration}: Requesting patch from GPT-4{' (with similar methods)' if include_similar else ''}...")
            try:
                responses = predictor.get_response(prompt, BEAM_SIZE)
            except Exception as api_error:
                print(f"OpenAI API call failed at iteration {iteration} for {bug_id}: {api_error}")
                break
            extracted_patches = []
            for patch in responses:
                extracted_patches.append(extracting_patches_from_output.extract_fixed_method(patch)[0])

            method['generation'][iteration]['patches'] = extracted_patches
        
        #generation for non direct methods 

        methods = [[method['comment'], method['buggy_function']] for method in non_direct_methods]
        if len(methods) > 0:
            failing_testcases = []
            for method in non_direct_methods:
                for test in method['non_directly_related_tests']:
                    failing_testcases.append(test)
            
            failing_tests = [bug_info['trigger_test'][test]  for test in bug_info['trigger_test'] if test in failing_testcases]     
            if len(failing_tests) != 0: 
                prompt = prompt_maker.build_prompt_mm(bug_id, methods, failing_tests, retrieved_context , include_similar)

                for method in non_direct_methods:
                    if 'generation' not in method:
                        method['generation'] = {}
                        method['generation'][iteration] = {} 
                        method['generation'][iteration]['prompt'] = prompt 
                    else:
                        method['generation'][iteration] = {}
                        method['generation'][iteration]['prompt'] = prompt
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
                        if 'patches' not in method['generation'][iteration]:
                            method['generation'][iteration]['patches'] = []
                        try:    
                            method['generation'][iteration]['patches'].append(extracted_patches[index])
                        except:
                            print("no extracted patches")
                        index+=1
        
        patches_saved[bug_id] = bug_info
        os.makedirs(save_dir, exist_ok=True)
        with open(save_file, 'w') as f:
            json.dump(patches_saved, f, indent=4)

import javalang

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
        patches = method['generation']['1']['patches']
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

            if 'plausible_patches' not in method['generation']['1']:
                method['generation']['1']['plausible_patches'] = []
            method['generation']['1']['plausible_patches'].append(patch)
        method['generation']['1']['evaluation_done'] = True
    if len(non_direct_methods) != 0:
        for i in range(10):
            for method in non_direct_methods:
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
                patches = method['generation']['1']['patches']
                if len(patches)>=i+1:
                    if 'plausible_patches' not in method['generation']['1']:
                        method['generation']['1']['plausible_patches'] = []
                    method['generation']['1']['plausible_patches'].append(patches[i])
            method['generation']['1']['evaluation_done'] = True
    patch_data[bug_id]['evaluation_done'] = True
    with open(f"{output_dir}/{bug_id.split('-')[0]}/{bug_id}.json","w") as file:
        json.dump(patch_data, file, indent=4)
    print(f"{bug_id} evaluation ended")

def process_methods_with_no_related_tests(bug_id_bug_info_tuple):
    import os
    import json
    import shutil

    # Re-import other required modules here if needed
    bug_id, bug_info = bug_id_bug_info_tuple


    project = bug_id.split('-')[0]
    bug_num = bug_id.split('-')[1] if '-' in bug_id else ""
    print(f"\n=== Processing {bug_id} ===")
    work_dir = os.path.join(output_dir, "checkouts", bug_id)
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    if bug_num:
        cmd = ["defects4j", "checkout", "-p", project, "-v", f"{bug_num}b", "-w", work_dir]
        ret, cout = run_defects4j_command(cmd, cwd=".")
        if ret != 0:
            print(f"Error: Failed to checkout {bug_id} - {cout.strip()}")
            return
        
    
    for iteration in range(1, 2):
        include_similar = (iteration > 1)
        save_dir = os.path.join(output_dir, f"{bug_id.split('-')[0]}")
        save_file = os.path.join(save_dir, f"{bug_id}.json")
        
        if os.path.exists(save_file):
            with open(save_file, 'r') as f:
                patches_saved = json.load(f)
        else:
            patches_saved = {}

        methods_without_testcase = [item for item in patches_saved[bug_id]['functions'] if 'no_relatable_failing_tests' in item] 
        other_methods = [item for item in patches_saved[bug_id]['functions'] if 'no_relatable_failing_tests' not in item]
        
        for method in methods_without_testcase:
            if 'generation' in method:
                return

        if include_similar:
            retrieved_context = query_vector_db.retrieve_context(bug_id)
        else:
            retrieved_context = []
        
        #looking for plausible patches

        bmethod_plausible_patch_pairs = []
        for method in other_methods:
                if 'generation' in method:
                    if str(iteration) in method['generation']:
                        if 'plausible_patches' in method['generation'][str(iteration)]:
                            bmethod_plausible_patch_pairs.append((method['buggy_function'], method['generation'][str(iteration)]['plausible_patches'][0]))
            
        if len(bmethod_plausible_patch_pairs) > 0:
            buggy_methods = [method['buggy_function'] for method in methods_without_testcase]
            prompt = prompt_maker.build_prompt_for_unlinked_buggy_methods(buggy_methods, bmethod_plausible_patch_pairs)
            try:
                print(f"Iteration {iteration}: Requesting patch from GPT-4{' (with similar methods)' if include_similar else ''}...")
                responses = predictor.get_response(prompt, BEAM_SIZE)
            except Exception as api_error:
                print(f"OpenAI API call failed at iteration {iteration} for {bug_id}: {api_error}")
                break
            extracted_patches = []
            for patch in responses:
                extracted_patches.append(extracting_patches_from_output.extract_fixed_method(patch))
            index = 0
            for method in methods_without_testcase:       
                if 'generation' not in method:
                    method['generation'] = {}
                    method['generation'][str(iteration)] = {} 
                    method['generation'][str(iteration)]['prompt'] = prompt 
                else:
                    if str(iteration) not in method['generation']:
                        method['generation'][str(iteration)] = {}
                    method['generation'][str(iteration)]['prompt'] = prompt                
                method['generation'][str(iteration)]['patches'] = [patch[index] for patch in extracted_patches if len(patch)>=index+1]
                index+=1
            patches_saved[bug_id]['generation_done_for_notest_methods'] = True         
        with open(save_file, 'w') as f:
            json.dump(patches_saved, f, indent=4)

# Outside the function
if __name__ == "__main__":

    # for bug in to_process.items():
    #     b,i=bug
    #     process_methods_with_no_related_tests(bug)

    # max_workers = multiprocessing.cpu_count()  # Use all available cores
    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     executor.map(process_bug, to_process.items())

    max_workers = multiprocessing.cpu_count()  # Use all available cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_methods_with_no_related_tests, to_process.items())
