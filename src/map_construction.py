import lib
import subprocess
import json
import re
import shutil
import javalang
import os   
import traceback
from multiprocessing import Pool, cpu_count

checked_out_projects_path = ""
import_file_path = ""
export_file_path = ""
error_log_path = ""



def run_tests(work_dir):
    command = f"defects4j test -w {work_dir}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stderr

def checkout_project(project_id, bug_id, work_dir):
    """Checks out the project from Defects4J."""
    command = f"defects4j checkout -p {project_id} -v {bug_id}f -w {work_dir}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error during checkout: {result.stderr}")
        return False
    print(f"Checked out project {project_id} (bug {bug_id}) to {work_dir}")
    return True


def copy_java_file(source_path, destination_path):
    try:
        shutil.copy(source_path, destination_path)
        print(f"File copied successfully from {source_path} to {destination_path}")
    except FileNotFoundError:
        print(f"Source file not found: {source_path}")
    except PermissionError:
        print(f"Permission denied while copying to: {destination_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def replace_method_auto(bug_id, method_json, to_which):
    # Step 1: Read the Java file
    java_file_path = work_dir = f"{checked_out_projects_path}/{bug_id}/{method_json['path']}"
    with open(java_file_path, 'r') as file:
        java_code = file.read()
    updated_code = ""
    if to_which == "to_fixed":
        if method_json['buggy_function'].strip() in java_code:
            if method_json['fixed_function'] is None:
                updated_code = java_code.replace(method_json['buggy_function'].strip(), "\n")
            else:
                updated_code = java_code.replace(method_json['buggy_function'].strip(), method_json['fixed_function'].strip())
        else:
            java_lines = java_code.splitlines()
            buggy_method_start = method_json['start_loc']
            buggy_method_end = method_json['end_loc']

            updated_code = (
                java_lines[:buggy_method_start]
                + method_json['fixed_function']
                + java_lines[buggy_method_end+1:]
            )
            updated_code= "\n".join(updated_code)
    elif to_which == "to_buggy":
        if method_json['fixed_function'] is not None:
            updated_code = java_code.replace(method_json['fixed_function'].strip(), method_json['buggy_function'].strip())
        else:
            java_lines = java_code.splitlines()
            updated_code = (
                java_lines[:method_json['start_loc']]
                + method_json['buggy_function'].splitlines()
                + java_lines[method_json['start_loc']:]
            )
            updated_code= "\n".join(updated_code)
    # Step 6: Write the updated code back to the file
    with open(java_file_path, 'w') as file:
        file.write(updated_code)
    print(f"{bug_id}---method replacement done")

def extract_failed_test_cases(log):
    failure_pattern = re.compile(r"--- ([\w\.]+)::([\w\d_]+)")
    failed_tests = failure_pattern.findall(log)
    failed_test_cases = [f"{suite}::{test}" for suite, test in failed_tests] 
    return failed_test_cases


def process_bug(bug_id_data):
    bug_id, bug_data = bug_id_data
    try:
        if "functions" not in bug_data:
            return None, None

        bug_project, bug_num = bug_id.split('-')
        work_dir = f"{checked_out_projects_path}/{bug_id}"

        lib.checkout_project_buggy(bug_project, bug_num, work_dir)
        run_tests(work_dir)

        with open(f"{work_dir}/failing_tests", 'r') as file:
            log = file.read()
        all_failing_tests = extract_failed_test_cases(log)

        directly_related_tcs_list = []

        for method in bug_data['functions']:
            replace_method_auto(bug_id, method, "to_fixed")
            run_tests(work_dir)

            if os.path.exists(f"{work_dir}/failing_tests"):
                with open(f"{work_dir}/failing_tests", 'r') as file:
                    log = file.read()
            else:
                method['Compiling error!'] = True
                method['non_directly_related_tests'] = []
                replace_method_auto(bug_id, method, "to_buggy")
                continue

            current_failed_test_cases = extract_failed_test_cases(log)
            related_test_cases = [item for item in all_failing_tests if item not in current_failed_test_cases]

            if related_test_cases:
                method['directly_related_tests'] = related_test_cases
                directly_related_tcs_list += related_test_cases
            else:
                method['non_directly_related_tests'] = []

            replace_method_auto(bug_id, method, "to_buggy")

        left_test_cases = [item for item in all_failing_tests if item not in directly_related_tcs_list]
        
        methods_to_replace = sorted(
            [m for m in bug_data['functions'] if 'non_directly_related_tests' in m],
            key=lambda x: x['start_loc'],
            reverse=True
        )

        for method in methods_to_replace:
            replace_method_auto(bug_id, method, "to_fixed")
        
        run_tests(work_dir)
        if not os.path.exists(f"{work_dir}/failing_tests"):
            for method in bug_data['functions']:
                if 'non_directly_related_tests' in method:
                    method['non_directly_related_tests'] = left_test_cases 
            return bug_id, bug_data
               
        with open(f"{work_dir}/failing_tests", 'r') as file:
            log = file.read()
            
        current_failed_test_cases = extract_failed_test_cases(log)

        if set(current_failed_test_cases) == set(directly_related_tcs_list):
            for method in bug_data['functions']:
                if 'Compiling error!' in method:
                    continue
                if 'non_directly_related_tests' in method:
                    replace_method_auto(bug_id, method, "to_buggy")
                    run_tests(work_dir)
                    if not os.path.exists(f"{work_dir}/failing_tests"):
                        replace_method_auto(bug_id, method, "to_fixed")
                        continue
                    with open(f"{work_dir}/failing_tests", 'r') as file:
                        log = file.read()
                    if set(extract_failed_test_cases(log)) == set(current_failed_test_cases):
                        method['no_relatable_failing_tests'] = True
                    replace_method_auto(bug_id, method, "to_fixed")
            for method in bug_data['functions']:
                if 'non_directly_related_tests' in method and 'no_relatable_failing_tests' not in method:
                    method['non_directly_related_tests'] = left_test_cases
        else:
            left_relatable_tests = [item for item in left_test_cases if item not in current_failed_test_cases]
            for method in bug_data['functions']:
                if 'non_directly_related_tests' in method:
                    method['non_directly_related_tests'] = left_relatable_tests if left_relatable_tests else [
                        item for item in all_failing_tests if item not in current_failed_test_cases
                    ]

        return bug_id, bug_data

    except Exception as e:
        error = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        return bug_id, error

def main():

    with open(import_file_path, "r") as file:
        data = json.load(file)

    if os.path.exists(export_file_path):
        with open(export_file_path, "r") as file:
            processed_bugs = json.load(file)
    else:
        processed_bugs = {}

    to_process = [(bug_id, data[bug_id]) for bug_id in data if "functions" in data[bug_id] and bug_id not in processed_bugs]
    
    error_log = {}
    for bug in to_process:
            bug = process_bug(bug)
            if "error_type" in bug[1]:
                with open(error_log_path, "w") as f:
                    json.dump({bug[0]:bug[1]}, f, indent=4)
            else:
                processed_bugs[bug[0]] = bug[1]
                with open(export_file_path, "w") as f:
                    json.dump(processed_bugs, f, indent=4)


    # print(f"Processing {len(to_process)} bugs in parallel...")

    # error_log = {}
    # results = []

    # with Pool(processes=cpu_count()) as pool:
    #     for bug_id, result in pool.map(process_bug, to_process):
    #         if bug_id is None:
    #             continue
    #         if isinstance(result, dict) and "error_type" in result:
    #             error_log[bug_id] = result
    #         else:
    #             processed_bugs[bug_id] = result

    #         # Save incrementally after each result
    #         with open(export_file_path, "w") as f:
    #             json.dump(processed_bugs, f, indent=4)
    #         with open(error_log_path, "w") as f:
    #             json.dump(error_log, f, indent=4)

    print("All done!")


if __name__ == "__main__":
    main()
