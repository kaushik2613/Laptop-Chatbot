import pandas as pd
import time
import random
import re
from playwright.sync_api import sync_playwright

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("models_test.csv")

results = []

PRIORITY_SITES = [
    "amazon.com",
    "bestbuy.com",
    "dell.com",
    "hp.com",
    "lenovo.com",
    "apple.com",
    "microsoft.com",
    "notebookcheck.net",
]

SPEC_KEYWORDS = [
    "RAM", "Memory", "GB", "TB",
    "Processor", "CPU", "Intel", "AMD", "Core i", "Ryzen", "Apple M",
    "Storage", "SSD", "HDD",
    "Display", "Screen", "Resolution", "FHD", "UHD", "OLED", "IPS",
    "Graphics", "GPU", "NVIDIA", "GeForce", "Radeon",
    "Battery", "Weight", "OS", "Windows", "macOS",
    "Hz", "Thunderbolt", "USB", "inch",
]


# =========================
# SEARCH VIA PLAYWRIGHT (Bing first, Google fallback)
# =========================
def search_bing(page, model):
    query = f"{model} laptop specifications"
    url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
    try:
        page.goto(url, timeout=30000, wait_until="domcontentloaded")
        page.wait_for_timeout(random.randint(1500, 2500))
        anchors = page.query_selector_all("li.b_algo h2 a, #b_results h2 a")
        links = [a.get_attribute("href") for a in anchors]
        links = [l for l in links if l and l.startswith("http")]
        print(f"  Bing found {len(links)} links")
        return links
    except Exception as e:
        print(f"  [Bing error] {e}")
        return []


def search_google(page, model):
    query = f"{model} laptop specifications"
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}&num=10"
    try:
        page.goto(url, timeout=30000, wait_until="domcontentloaded")
        page.wait_for_timeout(random.randint(1500, 2500))
        if "sorry/index" in page.url or "captcha" in page.content().lower():
            print("  [Google] CAPTCHA detected")
            return []
        anchors = page.query_selector_all("div#search a[href]")
        links = [a.get_attribute("href") for a in anchors]
        links = [l for l in links if l and l.startswith("http") and "google.com" not in l]
        print(f"  Google found {len(links)} links")
        return links
    except Exception as e:
        print(f"  [Google error] {e}")
        return []


def pick_best_link(links):
    scored = []
    for link in links:
        score = 0
        for i, site in enumerate(PRIORITY_SITES):
            if site in link:
                score = len(PRIORITY_SITES) - i
                break
        if any(x in link for x in ["/dp/", "/p/", "/product", "/laptop", "/pdp/"]):
            score += 2
        if any(x in link for x in ["/s?", "/search?", "/category", "/browse"]):
            score -= 1
        if score > 0:
            scored.append((score, link))
    scored.sort(reverse=True)
    if scored:
        return scored[0][1]
    for link in links:
        if any(site in link for site in PRIORITY_SITES):
            return link
    return links[0] if links else None


def get_product_link(page, model):
    links = search_bing(page, model)
    if not links:
        links = search_google(page, model)
    if not links:
        print("  No links found.")
        return None
    return pick_best_link(links)


# =========================
# SITE-SPECIFIC EXTRACTORS
# =========================
def extract_amazon(page):
    raw = []
    for el in page.query_selector_all("#feature-bullets li span.a-list-item"):
        t = el.inner_text().strip()
        if t:
            raw.append(t)
    for row in page.query_selector_all(
        "#productDetails_techSpec_section_1 tr, "
        "#productDetails_detailBullets_sections1 tr, "
        ".a-keyvalue tr"
    ):
        cells = row.query_selector_all("th, td")
        if len(cells) == 2:
            raw.append(f"{cells[0].inner_text().strip()}: {cells[1].inner_text().strip()}")
    return raw


def extract_bestbuy(page):
    raw = []
    for sel in [
        "button[data-lid='pdp-specifications-tab']",
        "button:has-text('Specifications')",
        "a[aria-label*='Specifications']",
    ]:
        try:
            btn = page.query_selector(sel)
            if btn and btn.is_visible():
                btn.click()
                page.wait_for_timeout(1500)
                break
        except:
            pass
    for el in page.query_selector_all(
        ".spec-table tr, .specification-row, "
        "[class*='spec-list'] li, [class*='SpecificationGroup'] li"
    ):
        t = el.inner_text().strip()
        if t and len(t) > 4:
            raw.append(t)
    return raw


def extract_dell(page):
    raw = []
    for row in page.query_selector_all(".techspecs-table tr, .ps-specs tr, [class*='spec'] tr"):
        cells = row.query_selector_all("td, th")
        if len(cells) >= 2:
            raw.append(f"{cells[0].inner_text().strip()}: {cells[1].inner_text().strip()}")
        elif len(cells) == 1:
            raw.append(cells[0].inner_text().strip())
    return raw


def extract_generic(page):
    raw = []
    try:
        text = page.inner_text("body")
        for line in text.split("\n"):
            line = line.strip()
            if 10 < len(line) < 300 and any(k.lower() in line.lower() for k in SPEC_KEYWORDS):
                raw.append(line)
    except:
        pass
    return raw[:50]


def deduplicate(lines):
    seen, out = set(), []
    for line in lines:
        norm = re.sub(r"\s+", " ", line).strip().lower()
        if norm not in seen and len(norm) > 5:
            seen.add(norm)
            out.append(line.strip())
    return out


def extract_specs(page, url):
    try:
        print(f"  Loading: {url[:90]}...")
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        page.wait_for_timeout(random.randint(2500, 4500))

        for sel in [
            "button#onetrust-accept-btn-handler",
            "button:has-text('Accept All')",
            "button:has-text('Accept Cookies')",
            "button:has-text('I Accept')",
        ]:
            try:
                btn = page.query_selector(sel)
                if btn and btn.is_visible():
                    btn.click()
                    page.wait_for_timeout(800)
                    break
            except:
                pass

        if "amazon.com" in url:
            raw = extract_amazon(page)
        elif "bestbuy.com" in url:
            raw = extract_bestbuy(page)
        elif "dell.com" in url:
            raw = extract_dell(page)
        else:
            raw = extract_generic(page)

        if len(raw) < 5:
            raw += extract_generic(page)

        cleaned = deduplicate(raw)
        return " | ".join(cleaned[:30]) if cleaned else None

    except Exception as e:
        print(f"  [Scrape error] {e}")
        return None


# =========================
# MAIN
# =========================
with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=False,
        args=["--disable-blink-features=AutomationControlled"]
    )
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        viewport={"width": 1280, "height": 800},
    )

    search_page = context.new_page()
    search_page.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    product_page = context.new_page()
    product_page.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )

    for model in df["model"]:
        if pd.isna(model):
            continue
        model = str(model).strip()
        print(f"\n{'='*55}")
        print(f"Processing: {model}")

        link = get_product_link(search_page, model)
        print(f"  Best link: {link}")

        specs = extract_specs(product_page, link) if link else None

        results.append({"model": model, "link": link, "specs": specs})
        print(f"  Specs: {'✅' if specs else '❌ None'}")
        time.sleep(random.uniform(4, 8))

    context.close()
    browser.close()

# =========================
# SAVE
# =========================
output_df = pd.DataFrame(results)
output_df.to_csv("output.csv", index=False)
print(f"\n✅ Done! output.csv — {len(output_df)} rows, {output_df['specs'].notna().sum()} with specs")