import re

def extract_rust_code(text):
    """
    Extracts Rust code blocks from text. Tries ```rust blocks first,
    then falls back to ``` blocks containing 'fn main'.
    
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
    
    # Fallback: look for any ``` block containing "fn main"
    generic_pattern = r"```\n?(.*?)```"
    generic_matches = re.finditer(generic_pattern, text, re.DOTALL)
    
    for match in generic_matches:
        code_block = match.group(1).strip()
        if "fn main" in code_block:
            return code_block
            
    return None