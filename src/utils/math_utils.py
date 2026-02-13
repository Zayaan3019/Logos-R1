"""
Math utilities for answer extraction and verification.
"""

import re
from typing import Optional, List, Union
import logging

logger = logging.getLogger(__name__)

# Optional sympy for symbolic math
try:
    from sympy import sympify, N, simplify
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logger.warning("sympy not available")


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numerical values from text.
    
    Args:
        text: Text to extract numbers from
    
    Returns:
        List of extracted numbers
    """
    # Pattern for numbers (including decimals, negatives, scientific notation)
    pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    
    return numbers


def extract_last_number(text: str) -> Optional[float]:
    """
    Extract the last number from text (common answer location).
    
    Args:
        text: Text to extract from
    
    Returns:
        Last number or None
    """
    numbers = extract_numbers(text)
    return numbers[-1] if numbers else None


def normalize_number(num: Union[str, float]) -> str:
    """
    Normalize a number for comparison.
    
    Handles:
    - Removing commas (1,234 -> 1234)
    - Removing currency symbols ($123 -> 123)
    - Converting to standard format
    
    Args:
        num: Number as string or float
    
    Returns:
        Normalized number string
    """
    if isinstance(num, (int, float)):
        # Remove trailing zeros from decimals
        s = f"{num:.10f}".rstrip('0').rstrip('.')
        return s
    
    # String processing
    s = str(num)
    
    # Remove common symbols
    s = re.sub(r'[,$\s]', '', s)
    
    # Convert to float and back to normalize
    try:
        val = float(s)
        # Remove trailing zeros
        s = f"{val:.10f}".rstrip('0').rstrip('.')
        return s
    except ValueError:
        return s


def numbers_close(num1: Union[str, float], num2: Union[str, float], tolerance: float = 1e-6) -> bool:
    """
    Check if two numbers are close (within tolerance).
    
    Args:
        num1: First number
        num2: Second number
        tolerance: Absolute tolerance
    
    Returns:
        True if numbers are close
    """
    try:
        if isinstance(num1, str):
            num1 = float(normalize_number(num1))
        if isinstance(num2, str):
            num2 = float(normalize_number(num2))
        
        return abs(float(num1) - float(num2)) < tolerance
    except (ValueError, TypeError):
        return False


def parse_mathematical_expression(expr: str) -> Optional[float]:
    """
    Parse and evaluate a mathematical expression.
    
    Args:
        expr: Mathematical expression string
    
    Returns:
        Evaluated result or None
    """
    if not SYMPY_AVAILABLE:
        # Fallback: try eval (dangerous, but limited scope)
        try:
            # Remove common text patterns
            expr = re.sub(r'[a-zA-Z]', '', expr)
            result = eval(expr, {"__builtins__": {}}, {})
            return float(result)
        except:
            return None
    
    try:
        # Use sympy for safe evaluation
        transformations = standard_transformations + (implicit_multiplication_application,)
        parsed = parse_expr(expr, transformations=transformations)
        result = N(parsed)
        return float(result)
    except:
        return None


def are_equivalent(expr1: str, expr2: str) -> bool:
    """
    Check if two mathematical expressions are equivalent.
    
    Args:
        expr1: First expression
        expr2: Second expression
    
    Returns:
        True if expressions are equivalent
    """
    if not SYMPY_AVAILABLE:
        # Fallback to numerical comparison
        val1 = parse_mathematical_expression(expr1)
        val2 = parse_mathematical_expression(expr2)
        
        if val1 is not None and val2 is not None:
            return numbers_close(val1, val2)
        
        # String comparison as last resort
        return normalize_number(expr1) == normalize_number(expr2)
    
    try:
        # Parse both expressions
        e1 = sympify(expr1)
        e2 = sympify(expr2)
        
        # Check if difference is zero
        diff = simplify(e1 - e2)
        
        if diff.is_number:
            return abs(float(N(diff))) < 1e-6
        
        return diff == 0
    except:
        # Fallback
        return expr1.strip() == expr2.strip()


def extract_answer_from_boxed(text: str) -> Optional[str]:
    """
    Extract answer from LaTeX boxed notation: \\boxed{answer}
    
    Args:
        text: Text potentially containing boxed answer
    
    Returns:
        Extracted answer or None
    """
    pattern = r'\\boxed\{([^}]+)\}'
    match = re.search(pattern, text)
    
    if match:
        return match.group(1).strip()
    
    return None


def extract_answer_patterns(text: str) -> Optional[str]:
    """
    Extract answer using multiple common patterns.
    
    Tries (in order):
    1. "The answer is X"
    2. "#### X" (GSM8K format)
    3. "Final answer: X"
    4. \\boxed{X}
    5. Last number in text
    
    Args:
        text: Text to extract answer from
    
    Returns:
        Extracted answer or None
    """
    # Pattern 1: "The answer is X"
    match = re.search(r'(?:the\s+)?answer\s+is:?\s*([+-]?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: GSM8K format "#### X"
    match = re.search(r'####\s*([+-]?\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1).strip()
    
    # Pattern 3: "Final answer: X"
    match = re.search(r'final\s+answer:?\s*([+-]?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern 4: Boxed notation
    boxed = extract_answer_from_boxed(text)
    if boxed:
        return boxed
    
    # Pattern 5: Last number
    last_num = extract_last_number(text)
    if last_num is not None:
        return str(last_num)
    
    return None


def format_number_for_display(num: Union[int, float]) -> str:
    """
    Format a number for human-readable display.
    
    Args:
        num: Number to format
    
    Returns:
        Formatted string
    """
    if isinstance(num, int):
        return f"{num:,}"
    
    # Float: show up to 4 decimal places, remove trailing zeros
    formatted = f"{num:.4f}".rstrip('0').rstrip('.')
    
    # Add commas for thousands
    parts = formatted.split('.')
    parts[0] = f"{int(parts[0]):,}"
    
    return '.'.join(parts) if len(parts) > 1 else parts[0]
