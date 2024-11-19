"""Code analysis tools."""

from __future__ import annotations

import ast
from typing import Any


async def analyze_code(code: str) -> dict[str, Any]:
    """Analyze Python code complexity and structure.

    Args:
        code: Python code to analyze

    Returns:
        Dictionary containing analysis metrics including:
        - Number of classes
        - Number of functions
        - Number of imports
        - Total lines of code
    """
    try:
        tree = ast.parse(code)
        return {
            "classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
            "functions": len([
                n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
            ]),
            "imports": len([n for n in ast.walk(tree) if isinstance(n, ast.Import)]),
            "lines": len(code.splitlines()),
        }
    except SyntaxError as exc:
        msg = f"Invalid Python code: {exc}"
        raise ValueError(msg) from exc


async def count_tokens(
    text: str,
    model: str = "gpt-4",
) -> dict[str, Any]:
    """Count the approximate number of tokens in text.

    Args:
        text: Text to analyze
        model: Model to use for tokenization (default: gpt-4)

    Returns:
        Dictionary containing:
        - token_count: Number of tokens
        - model: Model used for tokenization
    """
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
        token_count = len(encoding.encode(text))
    except Exception as exc:
        msg = f"Token counting failed: {exc}"
        raise ValueError(msg) from exc
    else:
        return {
            "token_count": token_count,
            "model": model,
        }
