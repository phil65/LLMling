from __future__ import annotations

from typing import Any, Literal

import playwright.async_api as pw

from llmling.tools.actions import ImmediateAction
from llmling.tools.base import LLMCallableTool


class BrowserTool(LLMCallableTool):
    """Tool for browser automation."""

    name = "browser"
    description = "Control a web browser to navigate and interact with web pages"

    def __init__(self) -> None:
        """Initialize browser tool."""
        self.browser: pw.Browser | None = None
        self.page: pw.Page | None = None

    async def startup(self) -> None:
        """Start browser instance."""
        playwright = await pw.async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=False)
        self.page = await self.browser.new_page()

    async def shutdown(self) -> None:
        """Close browser instance."""
        if self.browser:
            await self.browser.close()

    async def execute(
        self,
        action: Literal["open", "click", "type", "read", "screenshot"] = "open",
        url: str | None = None,
        selector: str | None = None,
        text: str | None = None,
        **kwargs: Any,
    ) -> ImmediateAction | dict[str, Any]:
        """Execute browser action."""
        # Create appropriate action based on command
        if action == "open":
            return ImmediateAction(
                action=self._open_page,
                args=(url,),
                kwargs={},
                requires_confirmation=True,
                description=f"Open URL: {url}",
            )

        if action == "click":
            return ImmediateAction(
                action=self._click_element,
                args=(selector,),
                kwargs={},
                requires_confirmation=True,
                description=f"Click element: {selector}",
            )

        if action == "type":
            return ImmediateAction(
                action=self._type_text,
                args=(selector, text),
                kwargs={},
                requires_confirmation=True,
                description=f"Type '{text}' into {selector}",
            )

        if action == "read":
            return ImmediateAction(
                action=self._read_element,
                args=(selector,),
                kwargs={},
                requires_confirmation=False,
                description=f"Read content from {selector}",
            )

        if action == "screenshot":
            return ImmediateAction(
                action=self._take_screenshot,
                args=(kwargs.get("path", "screenshot.png"),),
                kwargs={},
                requires_confirmation=True,
                description="Take screenshot of current page",
            )

        msg = f"Unknown action: {action}"
        raise ValueError(msg)

    async def _open_page(self, url: str) -> dict[str, str]:
        """Open a web page."""
        if not self.page:
            msg = "Browser not initialized"
            raise RuntimeError(msg)

        await self.page.goto(url)
        return {"title": await self.page.title(), "url": self.page.url}

    async def _click_element(self, selector: str) -> dict[str, str]:
        """Click an element."""
        if not self.page:
            msg = "Browser not initialized"
            raise RuntimeError(msg)

        await self.page.click(selector)
        return {"clicked": selector}

    async def _type_text(self, selector: str, text: str) -> dict[str, str]:
        """Type text into an element."""
        if not self.page:
            msg = "Browser not initialized"
            raise RuntimeError(msg)

        await self.page.fill(selector, text)
        return {"typed": text, "into": selector}

    async def _read_element(self, selector: str) -> dict[str, str]:
        """Read text from an element."""
        if not self.page:
            msg = "Browser not initialized"
            raise RuntimeError(msg)

        if not (element := await self.page.query_selector(selector)):
            return {"error": f"Element not found: {selector}"}

        text = await element.text_content()
        return {"text": text or ""}

    async def _take_screenshot(self, path: str) -> dict[str, str]:
        """Take a screenshot."""
        if not self.page:
            msg = "Browser not initialized"
            raise RuntimeError(msg)

        await self.page.screenshot(path=path)
        return {"screenshot": path}


if __name__ == "__main__":
    import asyncio

    from llmling.client import LLMLingClient

    async def main():
        async with LLMLingClient(
            "src/llmling/resources/web_research.yml", log_level="DEBUG"
        ) as client:
            print("\nAvailable tools:", client.tool_registry.list_items())
            print("Tool registry contents:", client.tool_registry._items)

            result = await client.execute("web_research")
            print(result.content)

            # Or with streaming
            async for chunk in await client.execute("web_research", stream=True):
                print(chunk.content, end="", flush=True)

    asyncio.run(main())
