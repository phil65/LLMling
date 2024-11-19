from __future__ import annotations

import ast


async def failing_tool(text: str) -> str:
    """Tool that always fails."""
    msg = "Intentional failure"
    raise ValueError(msg)


async def example_tool(text: str, repeat: int = 1) -> str:
    """Example tool that repeats text.

    Args:
        text: Text to repeat
        repeat: Number of times to repeat

    Returns:
        The repeated text
    """
    return text * repeat


async def analyze_ast(code: str) -> dict[str, int]:
    """Analyze Python code AST.

    Args:
        code: Python source code to analyze

    Returns:
        Dictionary with analysis results
    """
    tree = ast.parse(code)
    return {
        "classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
        "functions": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
    }
