from __future__ import annotations

from typing import Any

from llmling.core import exceptions


async def render_template(content: str, **kwargs: Any) -> str:
    """Render content as a Jinja2 template.

    This processor uses the global Jinja2 environment from RuntimeConfig.

    Args:
        content: Template content to render
        **kwargs: Variables to pass to the template

    Returns:
        Rendered template content

    Raises:
        ProcessorError: If template rendering fails
    """
    try:
        # RuntimeConfig's template engine is injected via kwargs
        template_engine = kwargs.pop("template_engine", None)
        if not template_engine:
            msg = "No template engine provided"
            raise ValueError(msg)  # noqa: TRY301

        result = await template_engine.render(content, **kwargs)
    except Exception as exc:
        msg = f"Template rendering failed: {exc}"
        raise exceptions.ProcessorError(msg) from exc
    else:
        return result
