import subprocess
import re
import shutil


import subprocess

def run_tests(work_dir):
    command = f"defects4j test -w {work_dir}"
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout + "\n" + result.stderr
        return output
    except Exception as e:
        return f"Error: {str(e)}"


def checkout_project_buggy(project_id, bug_num, work_dir):
    """Checks out the project from Defects4J."""
    command = f"defects4j checkout -p {project_id} -v {bug_num}b -w {work_dir}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error during checkout: {result.stderr}")
        return False
    print(f"Checked out project {project_id} (bug {bug_num}) to {work_dir}")
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

def apply_patch(file_path, fromm, to, start=0, end=1):
    # Step 1: Read the Java file
    with open(file_path, 'r') as file:
        java_code = file.read()
    if fromm in java_code:
        updated_code = java_code.replace(fromm, to)
    else:
        java_lines = java_code.splitlines()
        buggy_method_start = start
        buggy_method_end = end

        updated_code = (
            java_lines[:buggy_method_start-1]
            + to.splitlines()
            + java_lines[buggy_method_end:]
        )
        updated_code= "\n".join(updated_code)
    # Step 6: Write the updated code back to the file
    with open(file_path, 'w') as file:
        file.write(updated_code)
    print(f"{file_path}--------------method replacement done")

def extract_failed_test_cases(log):
    failure_pattern = re.compile(r"--- ([\w\.]+)::([\w\d_]+)")
    failed_tests = failure_pattern.findall(log)
    failed_test_cases = [f"{suite}::{test}" for suite, test in failed_tests] 
    return failed_test_cases
