import asyncio
import json
from typing import List, Dict

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup


BASE_URL = "https://mcpmarket.com"
OFFICIAL_MCP_URL = f"{BASE_URL}/categories/official"
OUTPUT_FILE = "/scratch4/home/akrik/NTILC/data/mcp/official_mcp_market_tools.json"


def create_tool_card_url(relative_url: str) -> str:
    return f"{BASE_URL}{relative_url}"


async def scrape_tool_cards() -> List[Dict]:
    """Scrape all tool-card links from the official mcp page using infinite scroll."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print("Loading official mcp page...")
        await page.goto(OFFICIAL_MCP_URL)

        prev_count = 0
        same_count_rounds = 0
        MAX_SAME_COUNT_ROUNDS = 3  # stop if no new cards after 3 scrolls

        while True:
            # Scroll to bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1.0)  # allow lazy content to load

            curr_count = await page.locator('a[id^="tool-card-"]').count()
            print(f"Loaded {curr_count} tool cards...")

            if curr_count == prev_count:
                same_count_rounds += 1
            else:
                same_count_rounds = 0

            if same_count_rounds >= MAX_SAME_COUNT_ROUNDS:
                break

            prev_count = curr_count

        html = await page.content()
        await browser.close()

    soup = BeautifulSoup(html, "html.parser")
    tool_cards = soup.select('a[id^="tool-card-"]')

    results = []
    for a in tool_cards:
        tool_id = a.get("id").removeprefix("tool-card-")
        href = a.get("href")
        if not href:
            continue
        results.append({
            "tool_id": tool_id,
            "href": href,
            "url": create_tool_card_url(href),
        })

    return results


async def fetch_html_after_optional_tools_click(page, url: str) -> str:
    """Fetch page HTML, clicking the Tools tab if it exists, using the same browser page."""
    await page.goto(url)
    await page.wait_for_load_state("networkidle")

    locator = page.locator("text=Tools")
    if await locator.count() > 0:
        await locator.first.click()
        await page.wait_for_load_state("networkidle")

    return await page.content()


def parse_tools(html: str) -> List[Dict]:
    """Parse tool definitions from a tool page."""
    soup = BeautifulSoup(html, "html.parser")
    tools = []

    tool_sections = soup.select("div.pb-8.border-b")

    for section in tool_sections:
        title_el = section.select_one("h3.text-lg.font-semibold")
        if not title_el:
            continue

        tool_name = title_el.get_text(strip=True)

        desc_el = section.select_one("p.text-sm.text-muted-foreground")
        description = desc_el.get_text(strip=True) if desc_el else None

        parameters = []
        for param in section.select("div.border.border-border\\/50"):
            name_el = param.select_one("code")
            type_el = param.select_one("span.font-mono")
            optional_text = param.get_text()
            optional = "Optional" in optional_text

            if not name_el or not type_el:
                continue

            parameters.append({
                "name": name_el.get_text(strip=True),
                "type": type_el.get_text(strip=True),
                "optional": optional,
            })

        tools.append({
            "tool_name": tool_name,
            "description": description,
            "parameters": parameters,
        })

    return tools


async def main():
    print("Scraping MCP Market tool cards...")
    tool_cards = await scrape_tool_cards()
    print(f"Found {len(tool_cards)} tool cards")

    all_data = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for idx, card in enumerate(tool_cards, start=1):
            print(f"[{idx}/{len(tool_cards)}] Scraping {card['tool_id']}")
            try:
                html = await fetch_html_after_optional_tools_click(page, card["url"])
                tools = parse_tools(html)

                all_data.append({
                    "tool_id": card["tool_id"],
                    "url": card["url"],
                    "tools": tools,
                })
            except Exception as e:
                print(f"  ⚠️ Failed on {card['tool_id']}: {e}")

        await browser.close()

    print(f"Writing output to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
