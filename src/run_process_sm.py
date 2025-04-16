from ast import List
import os
import json
import re
import subprocess
import openai
import shutil
import logging
import javalang
from pydantic import InstanceOf

# Setup logging
logging.basicConfig(
    filename="evaluation.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
from lib import prompt_maker, predictor, extracting_patches_from_output, query_vector_db
from lib import evaluation, checkout_projects

# Configuration
json_path = ""  # path to the JSON file with bug data
output_dir = ""                     # base directory to store outputs per project
TOP_N_SIMILAR = 5                         # number of similar methods to include when needed
MAX_ITERATIONS = 10                        # maximum iterations for trying patches
BEAM_SIZE = 10  
current_iteration = 3                     

# Load bug data from JSON
with open(json_path, 'r') as f:
    bugs_data = json.load(f)
to_process = {}
for bug in bugs_data:
    bug_dir = os.path.join(output_dir, f"{bug.split('-')[0]}")
    bug_output_file = os.path.join(bug_dir, f"{bug}.json")
    if os.path.exists(bug_output_file):
        with open(bug_output_file, 'r') as file:
            saved_data = json.load(file)
            if '2' in saved_data[bug] and 'plausible_patches' not in saved_data[bug]['2']:
                to_process[bug] = bugs_data[bug]
    else:
        print(bug_output_file)

os.makedirs(output_dir, exist_ok=True)
# Set OpenAI API key from environment (ensure OPENAI_API_KEY is defined)

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

    current_iteration = '2'
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
            print(f"Error: Failed to checkout {bug_id} – {cout.strip()}")
            return

        include_similar = True
        bug_dir = os.path.join(output_dir, f"{bug_id.split('-')[0]}")
        bug_file = os.path.join(bug_dir, f"{bug_id}.json")

        if os.path.exists(bug_file):
            with open(bug_file, 'r') as f:
                patches_saved = json.load(f)
        else:
            patches_saved = {bug_id: {current_iteration: {}}}

        if include_similar:
            retrieved_context = query_vector_db.retrieve_context(bug_id, [bug_info['buggy']])
        else:
            retrieved_context = []
        retrieved_methods = [item[1] for item in retrieved_context]
        retrieved_methods = remove_duplicate_methods(retrieved_methods)
        failing_tests = [bug_info['trigger_test'][test] for test in bug_info['trigger_test']]
        prompt = prompt_maker.build_prompt_sm(bug_id, [bug_info['buggy_code_comment'], bug_info['buggy']], failing_tests, retrieved_context , include_similar)
        if current_iteration not in patches_saved[bug_id]:
            patches_saved[bug_id][current_iteration]={}
        patches_saved[bug_id][current_iteration]['prompt'] = prompt

        print(f"Iteration {current_iteration}: Requesting patch from GPT-4{' (with similar methods)' if include_similar else ''}...")
        try:
            responses = predictor.get_response(prompt, BEAM_SIZE)
        except Exception as api_error:
            print(f"OpenAI API call failed at iteration {current_iteration} for {bug_id}: {api_error}")

        extracted_patches = []
        for patch in responses:
            extracted_patches.append(extracting_patches_from_output.extract_fixed_method(patch))

        patches_saved[bug_id][current_iteration]['patches'] = extracted_patches
        patches_saved[bug_id][current_iteration]['regeneration'] = True

        with open(bug_file, 'w') as f:
            json.dump(patches_saved, f, indent=4)

def tokenize_java_method(method_code):
    try:
        if isinstance(method_code, list):
            method_code = method_code[0].replace('\\"', '"').replace("\\'", "'")
        else:
            method_code = method_code.replace('\\"', '"').replace("\\'", "'")
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
    project, bug_num = bug_id.split('-')

    iteration = '2'

    checkout_dir = ""
    output_dir = ""  # Ensure this is defined

    try:
        logging.info(f"Checking out buggy project: {bug_id}")
        evaluation.checkout_project_buggy(project, bug_num, f"{checkout_dir}/{bug_id}")
    except Exception as e:
        logging.error(f"Failed to checkout project {bug_id}: {e}")
        return

    try:
        with open(f"{output_dir}/{project}/{bug_id}.json", "r") as file:
            patch_data = json.load(file)
        with open("----", "r") as file:
            data = json.load(file)
    except Exception as e:
        logging.error(f"Error loading JSON data for {bug_id}: {e}")
        return
    patches = patch_data[bug_id][str(iteration)]['patches']
    patches = remove_duplicate_methods(patches)

    logging.info(f"{bug_id} evaluation started with {len(patches)} patches")
    print(f"{bug_id} evaluation started")
    comp = 0
    for idx, patch in enumerate(patches):
        try:
            logging.info(f"[{bug_id}] Applying patch {idx + 1}/{len(patches)}")
            evaluation.apply_patch(
                f"{checkout_dir}/{bug_id}/{data[bug_id]['loc']}",
                data[bug_id]['buggy'],
                patch[0],
                data[bug_id]['start'],
                data[bug_id]['end']
            )
            try:
                output = evaluation.run_tests(f"{checkout_dir}/{bug_id}")
            except:
                logging.error(f"[{bug_id}] Exception while running_test {idx + 1}: {e}")
            evaluation.apply_patch(
                    f"{checkout_dir}/{bug_id}/{data[bug_id]['loc']}",
                    patch[0],
                    data[bug_id]['buggy']
            )
            
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
                    failing_tests = extract_failed_test_cases_with_optional_errors(log)
                if len(failing_tests) != len(raw_failing_tests):
                    failing_tests = raw_failing_tests
            elif test_result['status'] == 'no_failures':
                patch_data[bug_id][str(iteration)].setdefault('plausible_patches', []).append(patch)                       

        except Exception as e:
            logging.error(f"[{bug_id}] Exception while processing patch {idx + 1}: {e}")
            comp += 1  # Count this as a compilation/test failure
    
    if comp == len(patches):
        logging.warning(f"{bug_id} - All patches failed to compile or test")
        patch_data[bug_id][str(iteration)]['compiler_error_or_timeout'] = True
    patch_data[bug_id][str(iteration)]['evaluation_done'] = True
    patch_data[bug_id][str(iteration)]['second_check'] = True
    try:
        with open(f"{output_dir}/{project}/{bug_id}.json", "w") as file:
            json.dump(patch_data, file, indent=4)
        logging.info(f"{bug_id} evaluation completed and results saved")
        print(f"{bug_id} evaluation ended")
    except Exception as e:
        logging.error(f"Failed to save updated results for {bug_id}: {e}")
def class_to_relpath(class_path):
    """Convert 'org.apache.commons.cli.ApplicationTest' → 'org/apache/commons/cli/ApplicationTest.java'"""
    parts = class_path.split(".")
    return os.path.join(*parts) + ".java"

def find_exact_java_file(root_dir, class_path):
    """Recursively find the Java file that matches the full class path exactly."""
    expected_rel_path = class_to_relpath(class_path)
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == expected_rel_path.split(os.sep)[-1]:  # quick filter
                full_path = os.path.join(dirpath, filename)
                # Check if the full relative path from root matches
                rel_path = os.path.relpath(full_path, root_dir)
                if rel_path.replace(os.sep, "/").endswith(expected_rel_path.replace(os.sep, "/")):
                    return full_path
    return None

def extract_method_from_file(file_path, method_name):
    """Extract a method by name from a Java source file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    method_pattern = re.compile(rf'\s*(public|private|protected)?\s*(static)?\s*[\w<>\[\]]+\s+{method_name}\s*\(')
    method_lines = []
    inside_method = False
    brace_count = 0

    for line in lines:
        if not inside_method and method_pattern.search(line):
            inside_method = True

        if inside_method:
            method_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0:
                break

    return ''.join(method_lines) if method_lines else None

def extract_method(project_root, method_signature):
    class_path, method_name = method_signature.split("::")
    java_file = find_exact_java_file(project_root, class_path)

    if not java_file:
        print(f"Java file for class '{class_path}' not found.")
        return None

    method_code = extract_method_from_file(java_file, method_name)
    return method_code

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

def feedback_process(bug_id_bug_info_tuple):
    bug_id, bug_info = bug_id_bug_info_tuple
    project, bug_num = bug_id.split('-')

    checkout_dir = "-----"
    output_dir = "----"  # Ensure this is defined

    try:
        logging.info(f"Checking out buggy project: {bug_id}")
        evaluation.checkout_project_buggy(project, bug_num, f"{checkout_dir}/{bug_id}")
    except Exception as e:
        logging.error(f"Failed to checkout project {bug_id}: {e}")
        return

    try:
        with open("/home/selab/Desktop/MF-dlt/resources/defects4j_sm.json", "r") as file:
            data = json.load(file)
    except Exception as e:
        logging.error(f"Error loading JSON data for {bug_id}: {e}")
        return

    failing_test_minimum  = len([testname for testname in bug_info['trigger_test']])

    for iteration in range(3, MAX_ITERATIONS):
        try:
            with open(f"{output_dir}/{project}/{bug_id}.json", "r") as file:
                patch_data = json.load(file)
        except Exception as e:
            logging.error(f"Error loading JSON data for {bug_id}: {e}")
            return
        if 'patches' in patch_data[bug_id][str(iteration-1)]:
            patches = patch_data[bug_id][str(iteration-1)]['patches']
        else:
            break
        patches = remove_duplicate_methods(patches)

        logging.info(f"{bug_id} evaluation started with {len(patches)} patches")
        print(f"{bug_id} evaluation started")
        comp = 0
        new_failing_tests = []

        for idx, patch in enumerate(patches):
            try:
                logging.info(f"[{bug_id}] Applying patch {idx + 1}/{len(patches)}")
                evaluation.apply_patch(
                    f"{checkout_dir}/{bug_id}/{data[bug_id]['loc']}",
                    data[bug_id]['buggy'],
                    patch[0],
                    data[bug_id]['start'],
                    data[bug_id]['end']
                )
                try:
                    output = evaluation.run_tests(f"{checkout_dir}/{bug_id}")
                    evaluation.apply_patch(
                        f"{checkout_dir}/{bug_id}/{data[bug_id]['loc']}",
                        patch[0],
                        data[bug_id]['buggy']
                )
                except:
                    evaluation.apply_patch(
                        f"{checkout_dir}/{bug_id}/{data[bug_id]['loc']}",
                        patch[0],
                        data[bug_id]['buggy']
                )
                
                # Revert the patch

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
            except Exception as e:
                logging.error(f"[{bug_id}] Exception while processing patch {idx + 1}: {e}")
                comp += 1  # Count this as a compilation/test failure
       
        if 'plausible_patches' in patch_data[bug_id][str(iteration-1)]:
            patch_data[bug_id]['iteration_done'] = True
            break      
        if comp == len(patches) and comp != 0:
            logging.warning(f"{bug_id} - All patches failed to compile or test")
            if str(iteration-1) not in patch_data[bug_id]:
                patch_data[bug_id][str(iteration-1)] = {}
            patch_data[bug_id][str(iteration-1)]['compiler_error'] = True
            patch_data[bug_id]['iteration_done'] = True
            break
        
        # prompt making and geneartion for the best patch

        if str(iteration) in patch_data[bug_id]:
            retrieved_context = query_vector_db.retrieve_context(bug_id, [patch_data[bug_id][str(iteration)]['best_patch_for_prompt']])
            retrieved_methods = [item[1] for item in retrieved_context]
            retrieved_methods = remove_duplicate_methods(retrieved_methods)
            failing_tests = []
            for test in patch_data[bug_id][str(iteration)]['remaining_failing_tests']:
                if test in bug_info['trigger_test']:
                      failing_tests.append(bug_info['trigger_test'][test])
                else:
                    test_src = extract_method(f"{checkout_dir}/{bug_id}", test)
                    for t in new_failing_tests:
                        if t['failing_test'] == test:
                            error_message = t['error_message']
                    failing_tests.append({"src": test_src, 'clean_error_msg': error_message})
            
            prompt = prompt_maker.build_prompt_feedback(bug_id, [[bug_info['buggy_code_comment'], bug_info['buggy']]], [patch_data[bug_id][str(iteration)]['best_patch_for_prompt']], failing_tests, retrieved_methods , True)
            patch_data[bug_id][str(iteration)]['prompt'] = prompt

            print(f"Iteration {iteration}: Requesting patch from GPT-4{' (with similar methods)' if True else ''}...")
            try:
                responses = predictor.get_response(prompt, BEAM_SIZE)
            except Exception as api_error:
                print(f"OpenAI API call failed at iteration {iteration} for {bug_id}: {api_error}")

            extracted_patches = []
            for patch in responses:
                extracted_patches.append(extracting_patches_from_output.extract_fixed_method(patch))

            patch_data[bug_id][str(iteration)]['patches'] = extracted_patches
            with open(f"{output_dir}/{project}/{bug_id}.json", "w") as file:
                json.dump(patch_data, file, indent=4)
        else:
            patch_data[bug_id]['iteration_done'] = True
            break
    try:
        patch_data[bug_id]['iteration_done'] = True
        with open(f"{output_dir}/{project}/{bug_id}.json", "w") as file:
            json.dump(patch_data, file, indent=4)
        logging.info(f"{bug_id} evaluation completed and results saved")
        print(f"{bug_id} evaluation ended")
    except Exception as e:
        logging.error(f"Failed to save updated results for {bug_id}: {e}")

# Outside the function
if __name__ == "__main__":

    for bug in to_process:
         process_bug((bug, bugs_data[bug]))
        # feedback_process((bug, to_process[bug]))
        #evaluate((bug, to_process[bug]))

    # max_workers = os.cpu_count()# Use all available cores
    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     executor.map(evaluate, to_process.items())
