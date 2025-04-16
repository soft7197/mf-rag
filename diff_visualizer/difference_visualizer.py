import json
import os
import difflib
import webbrowser
from pathlib import Path
import subprocess


WORK_DIR = "/home/selab/Desktop/MF-dlt/diff_visualizer"
# List of (bug_id, buggy_dir, fixed_dir) tuples


OUTPUT_DIR = "visual_diff_output_all"

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

def read_lines(path):
    with open(path, 'r', errors='ignore') as f:
        return f.readlines()

def find_java_files(root_dir):
    java_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".java"):
                full_path = os.path.join(root, file)
                relative = os.path.relpath(full_path, root_dir)
                java_files.append(relative)
    return java_files

def make_diff_html(bug_id, buggy_path, fixed_path, output_path, relative_name):
    buggy_lines = read_lines(buggy_path)
    fixed_lines = read_lines(fixed_path)

    differ = difflib.HtmlDiff(tabsize=4, wrapcolumn=80)
    html = differ.make_file(
        buggy_lines, fixed_lines,
        fromdesc=f"{bug_id} - Buggy: {relative_name}",
        todesc=f"{bug_id} - Fixed: {relative_name}",
        context=True, numlines=3
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

def compare_projects(bug_id, buggy_dir, fixed_dir, output_base_dir):
    java_files = find_java_files(buggy_dir)
    diffs_created = []

    for rel_path in java_files:
        buggy_path = os.path.join(buggy_dir, rel_path)
        fixed_path = os.path.join(fixed_dir, rel_path)

        if not os.path.exists(fixed_path):
            continue

        with open(buggy_path, 'r', errors='ignore') as f1, open(fixed_path, 'r', errors='ignore') as f2:
            if f1.read() == f2.read():
                continue

        safe_rel_path = rel_path.replace(os.sep, '_')
        output_path = os.path.join(output_base_dir, f"{bug_id}__{safe_rel_path}.html")
        make_diff_html(bug_id, buggy_path, fixed_path, output_path, rel_path)
        diffs_created.append((bug_id, rel_path, os.path.basename(output_path)))

    return diffs_created

def generate_index_html(all_diffs, output_dir):
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, 'w', encoding="utf-8") as f:
        f.write("""
<html>
<head>
    <meta charset="UTF-8">
    <title>Defects4J Visual Diff Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f8f8f8; }
        h1 { text-align: center; }
        .bug-section { margin-bottom: 20px; border: 1px solid #ccc; border-radius: 8px; background: #fff; }
        .bug-header { cursor: pointer; padding: 10px 20px; font-size: 18px; font-weight: bold; background: #e6e6e6; border-radius: 8px 8px 0 0; }
        .bug-body { display: none; padding: 10px 20px; }
        ul { list-style-type: none; padding-left: 0; }
        li { margin: 6px 0; }
        a { text-decoration: none; color: #007acc; }
        a:hover { text-decoration: underline; }
    </style>
    <script>
        function toggleBody(id) {
            const body = document.getElementById(id);
            body.style.display = body.style.display === "none" ? "block" : "none";
        }
    </script>
</head>
<body>
    <h1>Defects4J Visual Diff Report</h1>
""")

        # Group diffs by bug_id
        bugs = {}
        for bug_id, rel_path, diff_file in all_diffs:
            bugs.setdefault(bug_id, []).append((rel_path, diff_file))

        for i, (bug_id, files) in enumerate(bugs.items()):
            section_id = f"bug_{i}"
            f.write(f"""
    <div class="bug-section">
        <div class="bug-header" onclick="toggleBody('{section_id}')">{bug_id}</div>
        <div class="bug-body" id="{section_id}">
            <ul>
""")
            for rel_path, diff_file in files:
                f.write(f'<li><a href="{diff_file}" target="_blank">{rel_path}</a></li>\n')
            f.write("""
            </ul>
        </div>
    </div>
""")

        f.write("</body></html>")
    return index_path


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    all_diffs = []
    with open("/home/selab/Desktop/MF-dlt/resources/defects4j_full.json", "r") as file:
        full = json.load(file)
    # for bug in full:
    #     checkout_project_buggy(bug.split("-")[0], bug.split("-")[1], f"{WORK_DIR}/checkouts/{bug.split('-')[0]}/{bug}-buggy")
    #     checkout_project_fixed(bug.split("-")[0], bug.split("-")[1], f"{WORK_DIR}/checkouts/{bug.split('-')[0]}/{bug}-fixed")
    bugs = sorted(
            [m for m in full],
            reverse=False
        )
    for bug in bugs:
        buggy_dir=f"{WORK_DIR}/checkouts/{bug.split('-')[0]}/{bug}-buggy"
        fixed_dir=f"{WORK_DIR}/checkouts/{bug.split('-')[0]}/{bug}-fixed"
        diffs = compare_projects(bug, buggy_dir, fixed_dir, OUTPUT_DIR)
        all_diffs.extend(diffs)

    if not all_diffs:
        print("No differences found across all bugs.")
        return

    index_file = generate_index_html(all_diffs, OUTPUT_DIR)
    print(f"✅ All done! Open index file: {index_file}")
    webbrowser.open(f"file://{os.path.abspath(index_file)}")

if __name__ == "__main__":
    main()
