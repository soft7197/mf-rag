
import json
import os
import re
from pathlib import Path

checkout_dir = "/home/selab/Desktop/MF-dlt/resources/checkouts"
def extract_main_error_msg(error_msg: str) -> str:
    return error_msg.strip().split("\n")[0].strip()
def extract_failed_test_line(clean_error_msg, expected_test_path):
    # Normalize expected path (file name only)
    test_file_name = expected_test_path.split('/')[-1]

    # Match lines like: at package.Class.method(File.java:123)
    pattern = re.findall(r'at [\w\.$]+\(.*?(\w+\.java):(\d+)\)', clean_error_msg)

    for file_name, line_number in pattern:
        if file_name == test_file_name:
            return int(line_number)
    return None
def extract_test_method_with_comment(file_path: str, fail_line: int) -> str:
    """Extracts the full test method containing the failure and adds a comment after the failing line."""
    with open(file_path, 'r', encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    method_start = None
    method_indent = None
    method_pattern = re.compile(
        r'^\s*'                                              # leading whitespace
        r'(?:@\w+\s*)*'                                      # optional annotations
        r'(?:public|protected|private)?\s*'                  # optional visibility
        r'(?:static\s+|final\s+|synchronized\s+|abstract\s+)*'  # optional modifiers
        r'(?:[\w\<\>\[\]]+\s+)+'                             # return type (with generics/arrays)
        r'([a-zA-Z_]\w*)\s*'                                 # method name (captured)
        r'\([^\)]*\)\s*'                                     # parameter list
        r'(?:throws\s+[^{]+)?'                               # optional throws clause
        r'\s*\{?'                                            # optional open brace
    )
    # Find method start
    for i in range(fail_line - 1, -1, -1):
        if method_pattern.match(lines[i]):
            method_start = i
            method_indent = len(lines[i]) - len(lines[i].lstrip())
            break

    if method_start is None:
        raise ValueError("Failed to locate method declaration.")

    # Find method end by counting braces
    open_braces = 0
    method_end = None
    for i in range(method_start, len(lines)):
        open_braces += lines[i].count('{')
        open_braces -= lines[i].count('}')
        if open_braces == 0:
            method_end = i
            break

    if method_end is None:
        raise ValueError("Failed to locate method end.")

    # Insert comment after the failing line
    fail_idx = fail_line - 1
    indent = len(lines[fail_idx]) - len(lines[fail_idx].lstrip())
    lines.insert(fail_idx + 1, ' ' * indent + '// this is the failed line\n')

    return ''.join(lines[method_start:method_end + 2])  # include comment
def build_prompt_mm(bug_id, buggy_methods: str, failing_tests_json: str, retrieved_methods: list, include_context = False) -> str:
    prompt = f"""You are an expert Java developer. Below are buggy method(s) from a large Java project. These methods cause one or more test failures.
Your task is to fix the bugs in these methods. Use the provided test failure(s) """
    if (include_context):
        prompt+="and relevant context "
    prompt+=f"""to guide your reasoning.

---

## Buggy Methods

"""
    for method in buggy_methods:
        prompt+=f"""
```java
{method[0]}
{method[1]}
```
"""
    prompt+=f"""
---

## Failing Test Case(s)

"""
    test_num = 1
    for test in failing_tests_json:
        prompt += f""" 
#Test method {test_num}:
```java
{test['src']}
```
#Error message from the test method {test_num}: {extract_main_error_msg(test['clean_error_msg'])}
"""     
        test_num+=1
    if include_context:
        prompt+=f"""

## 💡 Context from Project
"""
        for method in retrieved_methods:
            prompt += f"""
            
```java
{method[1]}
```
"""
    if len(buggy_methods)>1:
        prompt += """
---

## Your Goal

Fix the buggy methods. Return only the fixed Java methods. Do not include explanations or other text.
"""
    else:
        prompt += """
---

## Your Goal

Fix the buggy method. Return only the fixed Java method. Do not include explanations or other text.
"""
    return prompt
def build_prompt_sm(bug_id, buggy_method: str, failing_tests_json: str, retrieved_methods: list, include_context = False) -> str:
    prompt = f"""You are an expert Java developer. Below is a buggy method from a large Java project. This method causes one or more test failures.
Your task is to fix the bug in this method. Use the provided test failure(s) """
    if (include_context):
        prompt+="and relevant context "
    prompt+=f"""to guide your reasoning.

---

## Buggy Method

```java
{buggy_method[0]}
{buggy_method[1]}
```
---

## Failing Test Case(s)

"""
    test_num = 1
    for test in failing_tests_json:
        prompt += f""" 
#Test method {test_num}:
```java
{test['src']}
```
#Error message: {test['clean_error_msg']}
""" 
        test_num+=1    
    if include_context:
        prompt+=f"""

## 💡 Context from Project
"""
        for method in retrieved_methods:
            if isinstance(method, list):
                method_code = method[1]
            else:
                method_code = method 
            prompt += f"""
            
```java
{method_code}
```
"""
    prompt += """
---

## Your Goal

Fix the buggy method. Return only the fixed Java method. Do not include explanations or other text.
"""
    return prompt
def build_prompt_for_unlinked_buggy_methods(unlinked_buggy_methods: list, few_shot_examples: list) -> str:

    prompt = """You are an expert Java developer. Below are several buggy methods from a large Java project.
Although these methods are not directly linked to any failing test cases, we suspect they may contain bugs.
Your task is to fix these methods using your reasoning and by learning from the few-shot examples provided below.

---

## Few-shot Examples

"""
    for i, (buggy_code, fixed_code) in enumerate(few_shot_examples, 1):
        prompt += f"""
### Example {i} - Buggy:
```java
{buggy_code}
```

### Example {i} - Fixed:
```java
{fixed_code}
```
"""

    prompt += """

---

## Buggy Methods to Fix

Below are the methods suspected to be buggy. Apply your best judgment to fix them based on patterns from the examples.

"""
    for i, method in enumerate(unlinked_buggy_methods, 1):
        prompt += f"""
### Method {i}:
```java
{method}
```
"""
    if len(unlinked_buggy_methods) > 1:
        prompt += """

---

## 🌟 Your Goal

Fix all the buggy methods above. Return only the fixed Java methods in the same order. Do not include explanations, comments, or extra text.
"""
    else:
        prompt += """

---

## 🌟 Your Goal

Fix the buggy method. Return only the fixed Java method. Do not include explanations, comments, or extra text.
"""

    return prompt
def build_prompt_feedback(bug_id, original_buggy_methods: list, best_patch_methods: list, test_cases: dict, retrieved_methods: list, include_context=False) -> str:

    prompt = f'''You are an expert Java developer. Below are buggy method(s) from a large Java project, along with their best fixed versions based on earlier analysis.
Although the best fixed versions reduce some failures, they still result in failing test cases. Your task is to improve these fixed versions so that they pass the remaining failing test case(s).'''

    if include_context:
        prompt += ''' 
        
## Original Buggy Method(s)
'''

    for method in original_buggy_methods:
        prompt += f"""
```java
{method[0]}
{method[1]}
```
"""

    prompt += """
---

## Best Fixed Method(s)
"""

    for method in best_patch_methods:
        prompt += f"""
```java
{method[0]}
```
"""

    prompt += """
---

## Failing Test Case(s)
"""
    for i, test_case in enumerate(test_cases, 1):
        prompt += f"""
### Test Case {i}:
```java
{test_case['src']}
```

#Error message from test case {i}:
{extract_main_error_msg(test_case['clean_error_msg'])}
"""

    if include_context:
        prompt += """

---

## 💡 Context from Project
"""
        for method in retrieved_methods:
            prompt += f"""
```java
{method}
```
"""

    if len(best_patch_methods) > 1:
        prompt += """

---

## 🌟 Your Goal

Review and improve the fixed methods. Return only the improved Java methods in the same order. Do not include explanations or other text.
"""
    else:
        prompt += """

---

## 🌟 Your Goal

Review and improve the fixed method. Return only the improved Java method. Do not include explanations or other text.
"""

    return prompt
