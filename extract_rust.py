import re

def extract_rust_code(text):
    """
    Extracts Rust code blocks from text. Tries three approaches:
    1. ```rust blocks first
    2. Falls back to ``` blocks containing 'fn main'
    3. Finally checks for unclosed ```rust blocks containing 'fn' and 'main' ending with }
    
    Args:
        text (str): Text containing Rust code blocks
        
    Returns:
        str: First matching code block found, or None if no code block is found
    """
    # First try ```rust pattern
    rust_pattern = r"```rust\n(.*?)```"
    rust_match = re.search(rust_pattern, text, re.DOTALL)
    
    if rust_match:
        return rust_match.group(1).strip()
    
    # Fallback 1: look for any ``` block containing "fn main"
    generic_pattern = r"```\n?(.*?)```"
    generic_matches = re.finditer(generic_pattern, text, re.DOTALL)
    
    for match in generic_matches:
        code_block = match.group(1).strip()
        if "fn main" in code_block:
            return code_block
            
    # Fallback 2: look for unclosed ```rust blocks that contain Rust code
    unclosed_rust_pattern = r"```rust\n(.*$)"
    unclosed_match = re.search(unclosed_rust_pattern, text, re.DOTALL)
    
    if unclosed_match:
        code = unclosed_match.group(1).strip()
        # Verify it has fn, main, and ends with }
        if "fn" in code and "main" in code and code.rstrip().endswith("}"):
            return code
            
    return None