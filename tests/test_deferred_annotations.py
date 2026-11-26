"""Prompts from callables whose annotations name TYPE_CHECKING-only types (PEP 649)."""

from typing import TYPE_CHECKING, Any

from llmling.prompts.models import DynamicPrompt
from llmling.prompts.utils import extract_function_info


if TYPE_CHECKING:
    from decimal import Decimal  # only imported for type checkers


def price(amount: Decimal, currency: str = "EUR") -> str:
    """Format a price.

    Args:
        amount: The amount
        currency: The currency
    """
    return f"{amount} {currency}"


def test_from_callable_with_unresolvable_annotation() -> None:
    prompt = DynamicPrompt.from_callable(price)
    hints = {arg.name: arg.type_hint for arg in prompt.arguments}
    assert hints == {"amount": Any, "currency": str}


def test_extract_function_info_with_unresolvable_annotation() -> None:
    args, desc = extract_function_info(price)
    assert desc == "Format a price."
    assert {a.name: a.type_hint for a in args} == {"amount": Any, "currency": str}
