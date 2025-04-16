import subprocess
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def checkout_project_buggy(project_id, bug_id, work_dir):
    """Checks out the project from Defects4J."""
    command = f"defects4j checkout -p {project_id} -v {bug_id}b -w {work_dir}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error during checkout: {result.stderr}")
        return False
    print(f"Checked out project {project_id} (bug {bug_id}) to {work_dir}")
    return True

def checkout_project_fixed(project_id, bug_id, work_dir):
    """Checks out the project from Defects4J."""
    command = f"defects4j checkout -p {project_id} -v {bug_id}f -w {work_dir}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error during checkout: {result.stderr}")
        return False
    print(f"Checked out project {project_id} (bug {bug_id}) to {work_dir}")
    return True

def process_bug(bug_id, base_checkout_path="/home/selab/Desktop/MF-dlt/checkouts"):
    project, bug_num = bug_id.split('-')
    checkout_path = os.path.join(base_checkout_path, bug_id)
    checkout_project_buggy(project, bug_num, checkout_path)
    print(f"✔️ Processed: {bug_id}")
checkoutdir = "/home/selab/Desktop/MF-dlt/checkouts"
import json
import os
from multiprocessing import Pool, cpu_count

def worker(bug):
    project, bug_id = bug.split('-')
    checkout_project_buggy(project, bug_id, f"{checkoutdir}/{project}/{bug}")

if __name__ == "__main__":
    # Load JSON
    with open("/home/selab/Desktop/MF-dlt/resources/defects4j_full.json", "r") as file:
        data = json.load(file)

    # Set number of processes (default: number of CPU cores)
    num_processes = cpu_count()

    with Pool(processes=num_processes) as pool:
        pool.map(worker, data)

