import re
import json

from sympy import true

def extract_fixed_method(output):
    pattern = r"```java(.*?)```"
    matches = re.findall(pattern, output, re.DOTALL)  # re.DOTALL allows . to match newlines
    return matches

def get_method_name(str):
    pattern = r"\b(?:public|private|protected|static|final|synchronized|native|abstract|default|strictfp)*\s+" \
            r"(?:<[^>]+>\s+)?(?:[\w\[\]<>]+)\s+([a-zA-Z_$][a-zA-Z\d_$]*)\s*\("  
    match = re.search(pattern, str)
    if match:
        return match.group(1)
    return ""
