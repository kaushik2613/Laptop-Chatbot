"""
Laptop RAG Chatbot v9 — powered by Ollama + ChromaDB
New features:
  - Web search via DuckDuckGo (no API key needed)
  - Wikipedia lookup for tech terms and specs
  - General RAG: user can drop any text/URL as context
  - Tool routing: decides when to search web vs laptop DB vs answer from memory
Compatible with Gradio 6.11+

Install extra deps first:
  pip install ddgs requests beautifulsoup4
"""

import os, sys, re
from datetime import datetime
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import ollama
import gradio as gr

# web tools
try:
    import requests
    from bs4 import BeautifulSoup
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    print("[Warning] requests/beautifulsoup4 not installed — web search disabled")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CSV_PATH     = "final1_tiered.csv"
OLLAMA_MODEL = "llama3.2"
COLLECTION   = "laptops_v9"
TOP_K        = 5
EMBED_MODEL  = "all-MiniLM-L6-v2"
MAX_MEMORY   = 20

# ─────────────────────────────────────────────
# WEB SEARCH — DuckDuckGo HTML (no API key)
# ─────────────────────────────────────────────
def web_search(query, max_results=4):
    """
    Searches DuckDuckGo and returns a list of
    {title, url, snippet} dicts. No API key needed.
    """
    if not WEB_AVAILABLE:
        return []
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        url  = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        resp = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        for r in soup.find_all("div", class_="result", limit=max_results):
            title_tag   = r.find("a", class_="result__a")
            snippet_tag = r.find("a", class_="result__snippet")
            if title_tag:
                results.append({
                    "title":   title_tag.get_text(strip=True),
                    "url":     title_tag.get("href",""),
                    "snippet": snippet_tag.get_text(strip=True) if snippet_tag else "",
                })
        return results
    except Exception as e:
        print(f"[WebSearch] Error: {e}")
        return []


def fetch_page(url, max_chars=3000):
    """
    Fetches and extracts main text from a URL.
    Used when the user pastes a URL or we want full article content.
    """
    if not WEB_AVAILABLE:
        return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script/style
        for tag in soup(["script","style","nav","footer","header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        return text[:max_chars]
    except Exception as e:
        print(f"[FetchPage] Error: {e}")
        return ""


def wikipedia_summary(topic):
    """
    Gets a short Wikipedia summary for a tech term.
    Returns plain text or empty string if not found.
    """
    if not WEB_AVAILABLE:
        return ""
    try:
        # Use Wikipedia's REST API summary endpoint
        slug = topic.replace(" ", "_")
        url  = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        headers = {"User-Agent": "LaptopAdvisor/1.0 (educational project)"}
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("extract", "")[:1500]
        return ""
    except Exception as e:
        print(f"[Wikipedia] Error: {e}")
        return ""


def format_search_results(results):
    """Formats search results into a readable context string."""
    if not results:
        return "No web search results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] {r['title']}\n"
            f"    {r['snippet']}\n"
            f"    URL: {r['url']}"
        )
    return "\n\n".join(lines)

# ─────────────────────────────────────────────
# TOOL ROUTER — decides what tool to use
# ─────────────────────────────────────────────
def route_query(msg, mode):
    """
    Returns one of:
      'laptop_db'    — search the laptop database
      'web_search'   — search the web
      'wikipedia'    — look up a tech term on Wikipedia
      'url_fetch'    — user pasted a URL to read
      'comparison'   — compare previously shown laptops
      'followup'     — follow-up on previous results
      'general'      — general chat / answer from LLM memory
    """
    m = msg.lower().strip()

    # URL pasted
    if re.search(r"https?://\S+", msg):
        return "url_fetch"

    # Comparison
    if any(x in m for x in ["compare","vs","versus","difference between",
                              "option 1 and","option 2","option 3"]):
        return "comparison"

    # Wikipedia / tech explanation
    wiki_triggers = ["what is","what are","explain","how does","how do",
                     "tell me about","what's a","define","meaning of"]
    if any(x in m for x in wiki_triggers):
        # If it's about a specific tech term
        tech_terms = ["cpu","gpu","ram","ssd","nvme","ghz","tdp","pcie","vram",
                      "ddr","lpddr","oled","ips","tn","va","hz","refresh",
                      "ryzen","intel","nvidia","amd","arm","snapdragon",
                      "benchmark","cinebench","geekbench","passmark"]
        if any(t in m for t in tech_terms):
            return "wikipedia"
        return "web_search"  # general knowledge question

    # Web search triggers
    web_triggers = ["latest","new","2024","2025","released","review","benchmark",
                    "price","buy","where to","news","announced","upcoming",
                    "best laptop","top laptop","recommend","which brand",
                    "vs rtx","vs intel","how much","cost","available"]
    if any(x in m for x in web_triggers):
        return "web_search"

    # Short follow-up — answer from memory + laptop DB
    followup_kw = ["enough","is it","is this","which is","good for","ok for",
                   "fine for","suitable","work for","handle","better for",
                   "best for","would it","can it","does it","will it",
                   "how about","what about","more options","alternative",
                   "instead","upgrade","lighter","heavier","battery life",
                   "that one","those","the one","first","second","third"]
    if any(x in m for x in followup_kw) and mode == "done":
        return "followup"

    # Laptop-specific search in DB
    laptop_kw = ["laptop","notebook","gaming","coding","editing","machine learning",
                 "college","school","work","portable","lightweight","rtx","gtx",
                 "macbook","windows","ubuntu","16gb","32gb","512gb","1tb"]
    if any(x in m for x in laptop_kw):
        return "laptop_db"

    # General chat
    return "general"

# ─────────────────────────────────────────────
# GUIDED QUESTIONS
# ─────────────────────────────────────────────
GUIDED_QUESTIONS = [
    (
        "What are you mainly going to use this laptop for?\n\n"
        "  1️⃣  Gaming\n"
        "  2️⃣  Video editing or creative work\n"
        "  3️⃣  Coding or software development\n"
        "  4️⃣  School, college, or everyday tasks\n"
        "  5️⃣  Machine learning or AI work\n\n"
        "Type a number or describe in your own words!"
    ),
    (
        "How much RAM are you looking for?\n\n"
        "  1️⃣  8GB  — light use\n"
        "  2️⃣  16GB — sweet spot\n"
        "  3️⃣  32GB — heavy workloads\n"
        "  4️⃣  No preference\n"
    ),
    (
        "What screen size works best for you?\n\n"
        "  1️⃣  Small & portable (13–14 inch)\n"
        "  2️⃣  Mid-size (15–16 inch)\n"
        "  3️⃣  Large (17+ inch)\n"
        "  4️⃣  No preference\n"
    ),
    (
        "Do you need a dedicated GPU?\n\n"
        "  1️⃣  Yes — gaming, video editing, ML\n"
        "  2️⃣  No  — integrated is fine\n"
        "  3️⃣  Not sure\n"
    ),
    (
        "Any brand preference?\n\n"
        "  1️⃣  HP   2️⃣  Dell   3️⃣  Lenovo\n"
        "  4️⃣  ASUS  5️⃣  Apple  6️⃣  MSI/Gigabyte/Razer\n"
        "  7️⃣  No preference\n"
    ),
]

ACKNOWLEDGEMENTS = [
    {"1":"Nice! I'll find something with a strong GPU and CPU for gaming.",
     "2":"Creative work needs a good CPU and display. Got it!",
     "3":"A coder's machine! Solid CPU and plenty of RAM coming up.",
     "4":"Light and reliable — makes sense for everyday use.",
     "5":"ML needs serious hardware. Top-tier CPU and dedicated GPU. On it!",
     "default":"Got it! That helps me narrow things down."},
    {"1":"8GB is fine for light use.","2":"16GB — the sweet spot. Smart pick.",
     "3":"32GB for serious workloads. Got it.","4":"I'll match RAM to your use case.",
     "default":"Noted."},
    {"1":"Compact and portable — great choice!","2":"15–16 inch, the most popular size.",
     "3":"Big screen for gaming or creative work. Got it.","4":"I'll pick the right size.",
     "default":"Perfect."},
    {"1":"Dedicated GPU it is!","2":"Integrated works great for your needs.",
     "3":"I'll figure it out based on your use case.","default":"Got it."},
    {"1":"HP it is!","2":"Dell — great choice.","3":"Lenovo, known for quality.",
     "4":"ASUS makes impressive machines.","5":"MacBooks are great!",
     "6":"Gaming brand — excellent choice.","7":"No preference — I'll find the absolute best!",
     "default":"Noted!"},
]

QUESTION_MAPS = [
    {"1":"gaming high performance dedicated GPU RTX CPU tier 3",
     "2":"video editing creative strong CPU display GPU",
     "3":"coding development fast CPU 16GB RAM SSD",
     "4":"school college everyday lightweight portable battery",
     "5":"machine learning AI top tier CPU dedicated GPU 32GB RAM"},
    {"1":"8GB RAM","2":"16GB RAM","3":"32GB RAM","4":""},
    {"1":"13 inch 14 inch portable lightweight","2":"15 inch 16 inch",
     "3":"17 inch 18 inch large screen","4":""},
    {"1":"dedicated GPU gaming GPU tier 2 tier 3","2":"integrated GPU","3":""},
    {"1":"HP","2":"Dell","3":"Lenovo","4":"ASUS",
     "5":"Apple MacBook","6":"MSI Gigabyte Razer gaming","7":""},
]

QUICK_REPLIES = [
    "Best gaming laptop",
    "Lightweight for college",
    "Best for ML work",
    "Compare option 1 and 2",
    "What is GPU tier?",
    "Latest RTX laptops",
]

def get_ack(q_index, answer):
    acks = ACKNOWLEDGEMENTS[q_index]
    return acks.get(answer.strip(), acks.get("default","Got it!"))

def expand_answer(q_index, answer):
    m = QUESTION_MAPS[q_index]
    a = answer.strip()
    return m.get(a, a) if m.get(a) else a

# ─────────────────────────────────────────────
# DATA + VECTOR STORE
# ─────────────────────────────────────────────
def build_spec_string(row):
    parts = []
    parts.append(f"Brand: {row['Brand']}")
    parts.append(f"Processor: {row.get('Processor_full','N/A')}")
    cpu_tier  = int(row.get('cpu_tier', 1))
    cpu_label = {1:"Basic",2:"Mid-range",3:"High-end",4:"Top-tier"}.get(cpu_tier,"Mid-range")
    parts.append(f"CPU Tier: {cpu_tier} ({cpu_label})")
    parts.append(f"CPU Single-core: {row.get('b_singleScore','N/A')}")
    parts.append(f"CPU Multi-core: {row.get('b_multiScore','N/A')}")
    parts.append(f"RAM: {row['RAM_GB']}GB {row.get('RAM_type','')}")
    parts.append(f"Storage: {row['Storage_capacity_GB']}GB {row.get('Storage_type','')}")
    gpu_name   = str(row.get('Graphics_name','N/A'))
    gpu_tier   = int(row.get('gpu_tier', 0))
    gpu_label  = {0:"Integrated",1:"Entry",2:"Mid-range",3:"High-end"}.get(gpu_tier,"Integrated")
    integrated = row.get('Graphics_integreted', True)
    gpu_type   = "Integrated" if integrated else "Dedicated"
    gpu_gb     = row.get('Graphics_GB','')
    gpu_mem    = f" {gpu_gb}GB" if pd.notna(gpu_gb) and gpu_gb else ""
    parts.append(f"GPU: {gpu_name}{gpu_mem} ({gpu_type})")
    parts.append(f"GPU Tier: {gpu_tier} ({gpu_label})")
    parts.append(f"GPU 3D Score: {row.get('b_G3Dmark','N/A')}")
    size  = row.get('Display_size_inches','N/A')
    ppi   = row.get('ppi', 0)
    touch = "Touchscreen" if row.get('Touch_screen', False) else "Non-touch"
    parts.append(f"Display: {size} inch {float(ppi):.0f}ppi {touch}")
    parts.append(f"OS: {row.get('Operating_system','N/A')}")
    return " | ".join(parts)


def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")
    df = df.dropna(subset=["Name"])
    df['specs'] = df.apply(build_spec_string, axis=1)
    df['model'] = df['Name'].astype(str).str.strip()
    print(f"[Data] Loaded {len(df)} laptops")
    return df


def build_vectorstore(df):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection(COLLECTION)
    except:
        pass
    col = client.create_collection(
        name=COLLECTION, embedding_function=ef,
        metadata={"hnsw:space":"cosine"}
    )
    docs, ids, metas = [], [], []
    for i, row in df.iterrows():
        docs.append(f"Model: {row['model']}\nSpecs: {row['specs']}")
        ids.append(str(i))
        metas.append({
            "model":    row["model"],
            "brand":    str(row.get("Brand","")),
            "ram":      str(row.get("RAM_GB","")),
            "storage":  str(row.get("Storage_capacity_GB","")),
            "display":  str(row.get("Display_size_inches","")),
            "cpu_tier": str(int(row.get("cpu_tier",1))),
            "gpu_tier": str(int(row.get("gpu_tier",0))),
            "os":       str(row.get("Operating_system","")),
        })
    for start in range(0, len(docs), 500):
        col.add(
            documents=docs[start:start+500],
            ids=ids[start:start+500],
            metadatas=metas[start:start+500]
        )
        print(f"[VectorDB] Indexed {min(start+500, len(docs))}/{len(docs)}")
    print(f"[VectorDB] Ready — {col.count()} laptops")
    return col


def load_vectorstore():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path="./chroma_db")
    col = client.get_collection(name=COLLECTION, embedding_function=ef)
    print(f"[VectorDB] Loaded — {col.count()} laptops")
    return col


def retrieve(collection, query, k=TOP_K):
    res = collection.query(
        query_texts=[query], n_results=k,
        include=["documents","metadatas","distances"]
    )
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({
            "document": doc,
            "model":    meta.get("model",""),
            "brand":    meta.get("brand",""),
            "ram":      meta.get("ram",""),
            "storage":  meta.get("storage",""),
            "display":  meta.get("display",""),
            "cpu_tier": meta.get("cpu_tier","1"),
            "gpu_tier": meta.get("gpu_tier","0"),
            "os":       meta.get("os",""),
            "score":    round(1 - dist, 3),
        })
    return hits


def build_laptop_context(hits):
    ctx = ""
    for i, h in enumerate(hits, 1):
        ctx += (
            f"\n--- Laptop {i} ---\n"
            f"Score: {h['score']} | RAM: {h['ram']}GB | "
            f"Storage: {h['storage']}GB | Display: {h['display']}in | "
            f"CPU Tier: {h['cpu_tier']} | GPU Tier: {h['gpu_tier']} | OS: {h['os']}\n"
            f"{h['document']}\n"
        )
    return ctx

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a friendly, knowledgeable laptop advisor AND general tech assistant.

You have access to:
1. A database of 1020 real laptops with specs and performance tiers
2. Web search results (when provided)
3. Wikipedia summaries (when provided)
4. Full conversation memory

Capabilities:
- Recommend laptops from the database with detailed explanations
- Answer general tech questions using web search context
- Explain tech terms simply (RAM, GPU, CPU tiers, benchmarks)
- Compare laptops or specs
- Look up latest news, prices, benchmarks when web results are provided
- Answer follow-up questions using conversation memory

Tone: warm, natural, like a knowledgeable friend. Not robotic or overly formal.

CPU Tier: 1=Basic(i3/Ryzen3), 2=Mid(i5/Ryzen5), 3=High(i7/Ryzen7), 4=Top(i9/Ryzen9)
GPU Tier: 0=Integrated, 1=Entry, 2=Mid(RTX3060), 3=High(RTX3070+)

For laptop recommendations, always recommend 3 options and end with a "My Pick" summary.
For general questions, answer conversationally with the provided context.
If you don't have enough context to answer accurately, say so honestly."""

RECOMMENDATION_PROMPT = """You are a friendly laptop advisor.

Recommend exactly 3 laptops. For each:

Option [N] — [Name]
[Why it fits the user]

Specs:
- Processor: [name] (CPU Tier [X]/4) — [plain English]
- Graphics: [GPU] (GPU Tier [X]/3) — [capability]
- Memory: [X]GB — [adequate/ideal/overkill]
- Storage: [X]GB [type]
- Screen: [X] inch
- OS: [OS]

Worth knowing: [one trade-off]

---
My Pick: [which one and why in 2-3 casual sentences]

CPU: 1=Basic, 2=Mid, 3=High, 4=Top | GPU: 0=Integrated, 1=Entry, 2=Mid, 3=High"""


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p if isinstance(p, str) else p.get("text","")
            for p in content
        )
    return str(content)


def build_messages(system_prompt, history, user_content):
    msgs = [{"role":"system","content":system_prompt}]
    for turn in history[-MAX_MEMORY:]:
        if isinstance(turn, dict):
            msgs.append({
                "role":    turn["role"],
                "content": extract_text(turn["content"])
            })
    msgs.append({"role":"user","content":user_content})
    return msgs


def stream_llm(messages):
    """Streams tokens, yields full accumulated text each time."""
    try:
        full   = ""
        stream = ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True)
        for chunk in stream:
            token = chunk.get("message",{}).get("content","")
            if token:
                full += token
                yield full
        if not full:
            yield "I couldn't generate a response. Please try again."
    except Exception as e:
        yield (
            f"Something went wrong.\n\nError: {e}\n\n"
            f"Make sure Ollama is running: ollama pull {OLLAMA_MODEL}"
        )

# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
def export_chat(history):
    if not history:
        return None
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(os.getcwd(), f"laptop_chat_{ts}.txt")
    lines    = [
        f"Laptop Advisor Chat — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "=" * 60 + "\n\n"
    ]
    for turn in history:
        if isinstance(turn, dict):
            role    = "You" if turn["role"] == "user" else "Advisor"
            content = extract_text(turn["content"])
            lines.append(f"{role}:\n{content}\n\n{'─'*40}\n\n")
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return filepath

# ─────────────────────────────────────────────
# ONBOARDING STATE
# ─────────────────────────────────────────────
class OnboardingState:
    def __init__(self):
        self.answers  = []
        self.expanded = []
        self.q_index  = 0
        self.done     = False

    def current_question(self): return GUIDED_QUESTIONS[self.q_index]

    def answer(self, text):
        self.answers.append(text)
        self.expanded.append(expand_answer(self.q_index, text))
        self.q_index += 1
        if self.q_index >= len(GUIDED_QUESTIONS): self.done = True

    def last_ack(self): return get_ack(self.q_index - 1, self.answers[-1])
    def as_query(self): return " ".join(a for a in self.expanded if a)
    def summary(self):
        labels = ["Use case","RAM","Screen","GPU","Brand"]
        return "\n".join(f"  {labels[i]}: {a}" for i, a in enumerate(self.answers))


def to_msg(role, content): return {"role":role,"content":content}

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
def save_feedback(rating, comment, last_user, last_bot):
    """Append star rating + comment to feedback.csv"""
    import csv
    fp = "feedback.csv"
    new_file = not os.path.exists(fp)
    with open(fp, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp","stars","comment","user_message","bot_response"])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            rating,
            (comment or "")[:500],
            (last_user or "")[:500],
            (last_bot or "")[:1500],
        ])
    return fp


def load_feedback_stats():
    """Read feedback.csv and return summary stats for the dashboard."""
    import csv
    fp = "feedback.csv"
    if not os.path.exists(fp):
        return {"count":0, "avg":0.0, "dist":[0,0,0,0,0], "recent":[]}
    rows = []
    try:
        with open(fp, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    stars = int(r.get("stars", 0))
                    if 1 <= stars <= 5:
                        rows.append({"stars":stars, "comment":r.get("comment",""), "ts":r.get("timestamp","")})
                except (ValueError, TypeError):
                    continue
    except Exception:
        return {"count":0, "avg":0.0, "dist":[0,0,0,0,0], "recent":[]}
    if not rows:
        return {"count":0, "avg":0.0, "dist":[0,0,0,0,0], "recent":[]}
    total = len(rows)
    avg   = sum(r["stars"] for r in rows) / total
    dist  = [0,0,0,0,0]
    for r in rows:
        dist[r["stars"]-1] += 1
    return {"count":total, "avg":avg, "dist":dist, "recent":rows[-5:]}


def build_ui(collection):
    session = {"ob": OnboardingState(), "mode": "start"}

    FIRST_MSG = (
        f"Hey! I'm your laptop advisor and tech assistant, powered by local AI.\n\n"
        f"I have **{collection.count()} laptops** in my database plus web search, "
        f"Wikipedia, and URL reading.\n\n"
        f"**1** → Guide me through 5 questions\n"
        f"**2** → Just tell me what you need\n\n"
        f"Or ask me anything right now!"
    )

    def chat(user_msg, history):
        if not user_msg.strip():
            yield "", history
            return
        ob   = session["ob"]
        mode = session["mode"]
        tool = route_query(user_msg, mode)
        history = history + [to_msg("user", user_msg)]

        if mode == "start" and tool not in ("web_search","wikipedia","url_fetch","general"):
            c = user_msg.strip()
            if c == "1":
                session["mode"] = "guided"; session["ob"] = OnboardingState(); ob = session["ob"]
                bot = (f"Great! Let's find your perfect laptop.\n\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━\nQuestion 1 of {len(GUIDED_QUESTIONS)}\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━\n\n{GUIDED_QUESTIONS[0]}")
                history = history + [to_msg("assistant", bot)]; yield "", history; return
            elif c == "2":
                session["mode"] = "free"
                history = history + [to_msg("assistant","Of course! Describe what you need — use case, RAM, screen size, brand — or ask any tech question!")]; yield "", history; return
            else:
                session["mode"] = "free"

        if session["mode"] == "guided" and not ob.done:
            ob.answer(user_msg); ack = ob.last_ack()
            if not ob.done:
                bot = (f"{ack}\n\n━━━━━━━━━━━━━━━━━━━━━━\n"
                       f"Question {ob.q_index+1} of {len(GUIDED_QUESTIONS)}\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━\n\n{ob.current_question()}")
                history = history + [to_msg("assistant", bot)]; yield "", history; return
            else:
                hits = retrieve(collection, ob.as_query()); ctx = build_laptop_context(hits)
                msgs = build_messages(RECOMMENDATION_PROMPT, history[:-1],
                    f"User preferences:\n{ob.summary()}\n\nAvailable laptops:\n{ctx}\n\nGive 3 recommendations.")
                session["mode"] = "done"
                prefix = (f"{ack}\n\nSearching **{collection.count()} laptops**...\n\n"
                          f"━━━━━━━━━━━━━━━━━━━━━━\n**Your Top Matches**\n━━━━━━━━━━━━━━━━━━━━━━\n\n"
                          f"Based on:\n{ob.summary()}\n\n")
                history = history + [to_msg("assistant", prefix)]; ft = ""
                for ft in stream_llm(msgs): history[-1] = to_msg("assistant", prefix+ft); yield "", history
                history[-1] = to_msg("assistant", prefix+ft+"\n\n━━━━━━━━━━━━━━━━━━━━━━\nAsk me anything as a follow-up!")
                yield "", history; return

        if tool == "web_search":
            history = history + [to_msg("assistant","🔍 Searching the web...")]; yield "", history
            ctx = format_search_results(web_search(user_msg))
            msgs = build_messages(SYSTEM_PROMPT, history[:-1], f"Question: {user_msg}\n\nWeb results:\n{ctx}\n\nAnswer naturally.")
            ft = ""
            for ft in stream_llm(msgs): history[-1] = to_msg("assistant",ft); yield "", history
            return
        if tool == "wikipedia":
            history = history + [to_msg("assistant","📖 Looking that up...")]; yield "", history
            term = re.sub(r"(what is|what are|explain|how does|tell me about|what's a)","",user_msg,flags=re.IGNORECASE).strip()
            summary = wikipedia_summary(term)
            if not summary: summary = format_search_results(web_search(user_msg))
            msgs = build_messages(SYSTEM_PROMPT, history[:-1], f"Question: {user_msg}\n\nReference:\n{summary}\n\nExplain simply.")
            ft = ""
            for ft in stream_llm(msgs): history[-1] = to_msg("assistant",ft); yield "", history
            return
        if tool == "url_fetch":
            url = re.search(r"https?://\S+", user_msg).group(0)
            history = history + [to_msg("assistant",f"📄 Reading {url}...")]; yield "", history
            msgs = build_messages(SYSTEM_PROMPT, history[:-1], f"User: {user_msg}\n\nPage:\n{fetch_page(url)}\n\nAnswer based on this.")
            ft = ""
            for ft in stream_llm(msgs): history[-1] = to_msg("assistant",ft); yield "", history
            return

        hits = retrieve(collection, user_msg); ctx = build_laptop_context(hits)
        uc = (f"{user_msg}\n\nLaptops:\n{ctx}\n\nCompare clearly." if tool=="comparison"
              else f"{user_msg}\n\nRelevant laptops:\n{ctx}")
        msgs = build_messages(SYSTEM_PROMPT, history[:-1], uc)
        history = history + [to_msg("assistant","")]
        for ft in stream_llm(msgs): history[-1] = to_msg("assistant",ft); yield "", history

    def reset():
        session["ob"] = OnboardingState(); session["mode"] = "start"
        return [to_msg("assistant", FIRST_MSG)], ""

    def export(history):
        if not history: return gr.update(value=None, visible=False)
        fp = export_chat(history)
        return gr.update(value=fp, visible=True) if fp else gr.update(visible=False)

    # ── FEEDBACK ──────────────────────────────
    def submit_star(stars, comment, history):
        """Save a 1-5 star rating with optional comment"""
        last_user = ""
        last_bot  = ""
        for turn in reversed(history or []):
            if isinstance(turn, dict):
                if not last_bot and turn["role"] == "assistant":
                    last_bot = extract_text(turn["content"])
                elif not last_user and turn["role"] == "user":
                    last_user = extract_text(turn["content"])
                if last_bot and last_user:
                    break
        save_feedback(int(stars), comment, last_user, last_bot)
        # refresh dashboard html and clear comment
        return render_dashboard(), gr.update(value="", placeholder=f"✓ Thanks for your {stars}-star rating!")

    def render_dashboard():
        """Render the live dashboard HTML from feedback.csv stats."""
        s = load_feedback_stats()
        if s["count"] == 0:
            return """
            <div class="panel">
              <div class="panel-title">Feedback Dashboard</div>
              <div style="text-align:center;padding:20px 0;color:#9499c4;font-size:.82rem">
                No ratings yet — be the first to rate a response below!
              </div>
            </div>
            """
        avg     = s["avg"]
        count   = s["count"]
        dist    = s["dist"]
        max_d   = max(dist) or 1
        # build distribution bars (5 stars at top, 1 at bottom)
        bars = ""
        for stars_n in range(5, 0, -1):
            cnt = dist[stars_n-1]
            pct = (cnt / max_d) * 100
            bars += f"""
            <div style="display:flex;align-items:center;gap:8px;margin:4px 0">
              <span style="font-size:.7rem;color:#c4b5fd;width:24px;font-weight:600">{stars_n}★</span>
              <div style="flex:1;height:8px;background:rgba(255,255,255,.04);border-radius:4px;overflow:hidden">
                <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,#6366f1,#8b5cf6);border-radius:4px;transition:width .5s ease"></div>
              </div>
              <span style="font-size:.7rem;color:#9499c4;width:22px;text-align:right;font-variant-numeric:tabular-nums">{cnt}</span>
            </div>
            """
        # average rating stars (filled vs empty)
        avg_stars_html = ""
        for i in range(1, 6):
            if avg >= i:
                avg_stars_html += '<span style="color:#fbbf24">★</span>'
            elif avg >= i - 0.5:
                avg_stars_html += '<span style="color:#fbbf24">⯨</span>'
            else:
                avg_stars_html += '<span style="color:#3f3f5e">★</span>'

        # recent comments
        recent_html = ""
        for r in reversed(s["recent"]):
            if r.get("comment"):
                star_str = "★" * r["stars"] + "☆" * (5 - r["stars"])
                recent_html += f"""
                <div style="padding:7px 9px;background:rgba(99,102,241,.06);border-radius:8px;margin-bottom:6px;border-left:2px solid #6366f1">
                  <div style="font-size:.66rem;color:#fbbf24;letter-spacing:1px">{star_str}</div>
                  <div style="font-size:.74rem;color:#c0c4e0;margin-top:3px">"{r['comment'][:120]}"</div>
                </div>
                """
        if not recent_html:
            recent_html = '<div style="font-size:.72rem;color:#9499c4;padding:6px 0;font-style:italic">No comments yet</div>'

        return f"""
        <div class="panel">
          <div class="panel-title">Feedback Dashboard</div>
          <div style="text-align:center;padding:6px 0 12px">
            <div style="font-size:2.2rem;font-weight:800;line-height:1;
                 background:linear-gradient(90deg,#fbbf24,#f59e0b);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 background-clip:text">{avg:.1f}</div>
            <div style="font-size:1.1rem;letter-spacing:2px;margin:4px 0 6px">{avg_stars_html}</div>
            <div style="font-size:.68rem;color:#9499c4;text-transform:uppercase;letter-spacing:.8px">
              {count} {"rating" if count == 1 else "ratings"}
            </div>
          </div>
          <div style="border-top:1px solid rgba(99,102,241,.15);padding-top:10px;margin-top:6px">
            <div style="font-size:.62rem;color:#9499c4;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">
              Distribution
            </div>
            {bars}
          </div>
          <div style="border-top:1px solid rgba(99,102,241,.15);padding-top:10px;margin-top:10px">
            <div style="font-size:.62rem;color:#9499c4;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">
              Recent Comments
            </div>
            {recent_html}
          </div>
        </div>
        """

    # ─── CSS ──────────────────────────────────────────────────────────────────
    CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:    #08090f;
  --bg1:   #0d0f1a;
  --bg2:   #12152b;
  --bg3:   #171b35;
  --c1:    #6366f1;   /* indigo */
  --c2:    #8b5cf6;   /* violet */
  --c3:    #06b6d4;   /* cyan */
  --c4:    #10b981;   /* emerald */
  --c5:    #f59e0b;   /* amber */
  --tx:    #e8eaf6;
  --tx2:   #9499c4;
  --br:    rgba(99,102,241,0.15);
  --br2:   rgba(99,102,241,0.3);
  --r:     14px;
  --r-sm:  8px;
}

/* ── RESET + BASE ── */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
  background: var(--bg) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  color: var(--tx) !important;
  margin: 0 !important;
}
footer { display: none !important; }
.gap { gap: 16px !important; }

/* ── ANIMATED BACKGROUND ── */
.gradio-container::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 70% 60% at 5% 10%,  rgba(99,102,241,.07) 0%, transparent 55%),
    radial-gradient(ellipse 50% 70% at 95% 90%,  rgba(139,92,246,.08) 0%, transparent 55%),
    radial-gradient(ellipse 40% 40% at 50% 50%,  rgba(6,182,212,.03)  0%, transparent 60%);
  animation: bgshift 18s ease-in-out infinite alternate;
}
@keyframes bgshift {
  0%   { opacity: .7;  transform: scale(1);    }
  50%  { opacity: 1;   transform: scale(1.04); }
  100% { opacity: .8;  transform: scale(.98);  }
}
/* dot-grid */
.gradio-container::after {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: radial-gradient(circle, rgba(99,102,241,.18) 1px, transparent 1px);
  background-size: 36px 36px;
  mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 100%);
  animation: gridpulse 8s ease-in-out infinite alternate;
}
@keyframes gridpulse { from { opacity:.4 } to { opacity:.8 } }

/* ── ALL CONTENT ABOVE BG ── */
.gr-block, .gr-box, .gradio-row, .gradio-column, .wrap, .gap { position: relative; z-index: 1; }

/* ── TOP BAR ── */
#topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 20px 12px;
  border-bottom: 1px solid var(--br);
  position: relative; z-index: 2;
}
.topbar-logo {
  display: flex; align-items: center; gap: 12px;
}
.logo-orb {
  width: 38px; height: 38px; border-radius: 12px;
  background: linear-gradient(135deg, var(--c1), var(--c2));
  display: flex; align-items: center; justify-content: center;
  font-size: 19px;
  box-shadow: 0 0 20px rgba(99,102,241,.4);
  animation: orbglow 3s ease-in-out infinite alternate;
  flex-shrink: 0;
}
@keyframes orbglow {
  from { box-shadow: 0 0 16px rgba(99,102,241,.35); }
  to   { box-shadow: 0 0 32px rgba(139,92,246,.55), 0 0 60px rgba(99,102,241,.15); }
}
.logo-text h1 {
  font-size: 1.25rem; font-weight: 700; letter-spacing: -.3px;
  background: linear-gradient(90deg, #fff 0%, var(--c3) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  margin: 0;
}
.logo-text p { font-size: .7rem; color: var(--tx2); margin: 2px 0 0; letter-spacing: .3px; }

.topbar-badges { display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
  padding: 4px 11px; border-radius: 20px;
  font-size: .68rem; font-weight: 600; letter-spacing: .4px;
  border: 1px solid;
}
.badge-indigo { background:rgba(99,102,241,.12); border-color:rgba(99,102,241,.35); color:#a5b4fc; }
.badge-cyan   { background:rgba(6,182,212,.1);   border-color:rgba(6,182,212,.3);   color:#67e8f9; }
.badge-emerald{ background:rgba(16,185,129,.1);  border-color:rgba(16,185,129,.3);  color:#6ee7b7; }
.badge-amber  { background:rgba(245,158,11,.1);  border-color:rgba(245,158,11,.3);  color:#fcd34d; }

/* ── STAT CARDS ── */
.stat-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  padding: 16px 20px 4px;
}
.stat-card {
  background: linear-gradient(145deg, var(--bg1), var(--bg2));
  border: 1px solid var(--br);
  border-radius: var(--r);
  padding: 14px 16px;
  position: relative; overflow: hidden;
  transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
  animation: statIn .5s ease both;
}
.stat-card:hover {
  transform: translateY(-3px);
  border-color: var(--br2);
  box-shadow: 0 12px 40px rgba(0,0,0,.35);
}
.stat-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.stat-card:nth-child(1)::before { background: linear-gradient(90deg, var(--c1), transparent); }
.stat-card:nth-child(2)::before { background: linear-gradient(90deg, var(--c2), transparent); }
.stat-card:nth-child(3)::before { background: linear-gradient(90deg, var(--c4), transparent); }
.stat-card:nth-child(4)::before { background: linear-gradient(90deg, var(--c3), transparent); }
.stat-card:nth-child(1) { animation-delay: .05s }
.stat-card:nth-child(2) { animation-delay: .1s  }
.stat-card:nth-child(3) { animation-delay: .15s }
.stat-card:nth-child(4) { animation-delay: .2s  }
@keyframes statIn {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
.stat-icon { font-size: 1.4rem; margin-bottom: 6px; }
.stat-num {
  font-size: 1.8rem; font-weight: 700; line-height: 1;
  font-variant-numeric: tabular-nums;
}
.stat-card:nth-child(1) .stat-num { color: var(--c1); }
.stat-card:nth-child(2) .stat-num { color: var(--c2); }
.stat-card:nth-child(3) .stat-num { color: var(--c4); }
.stat-card:nth-child(4) .stat-num { color: var(--c3); }
.stat-label {
  font-size: .65rem; color: var(--tx2);
  text-transform: uppercase; letter-spacing: .8px; margin-top: 4px;
}

/* ── SIDEBAR PANELS ── */
.panel {
  background: linear-gradient(145deg, var(--bg1), var(--bg2));
  border: 1px solid var(--br);
  border-radius: var(--r);
  padding: 16px;
  margin-bottom: 12px;
  transition: border-color .25s;
}
.panel:hover { border-color: var(--br2); }
.panel-title {
  font-size: .65rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.2px;
  color: var(--tx2);
  border-bottom: 1px solid var(--br);
  padding-bottom: 8px; margin-bottom: 12px;
}

/* ── TIER ROWS ── */
.tier-row {
  display: flex; align-items: center; gap: 10px;
  padding: 7px 9px; border-radius: var(--r-sm);
  background: rgba(255,255,255,.025);
  border: 1px solid transparent;
  transition: all .2s; margin-bottom: 6px; cursor: default;
}
.tier-row:last-child { margin-bottom: 0; }
.tier-row:hover { background: rgba(99,102,241,.08); border-color: rgba(99,102,241,.2); }
.tier-badge {
  width: 26px; height: 26px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: .72rem; font-weight: 700; flex-shrink: 0;
  font-family: 'JetBrains Mono', monospace;
}
.tier-info .tier-name { font-size: .8rem; font-weight: 600; }
.tier-info .tier-chip { font-size: .66rem; color: var(--tx2); margin-top: 1px; }

/* ── CAPABILITY ROWS ── */
.cap-row {
  display: flex; align-items: center; gap: 9px;
  font-size: .78rem; color: #94a3b8;
  padding: 5px 0;
}
.cap-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--c4); margin-left: auto; flex-shrink: 0;
  animation: dotpulse 2s ease-in-out infinite;
}
@keyframes dotpulse {
  0%,100% { opacity: 1;   transform: scale(1);   }
  50%      { opacity: .5; transform: scale(1.6); }
}

/* ── CHATBOT PANEL ── */
.chat-wrap {
  background: linear-gradient(145deg, var(--bg1), var(--bg2));
  border: 1px solid var(--br);
  border-radius: var(--r);
  overflow: hidden;
  display: flex; flex-direction: column;
  box-shadow: 0 8px 48px rgba(0,0,0,.4);
}

/* chat header */
.chat-header {
  padding: 14px 18px;
  border-bottom: 1px solid var(--br);
  display: flex; align-items: center; gap: 12px;
  background: rgba(255,255,255,.02);
}
.chat-avatar {
  width: 36px; height: 36px; border-radius: 50%;
  background: linear-gradient(135deg, var(--c1), var(--c2));
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; flex-shrink: 0; position: relative;
  animation: avataurglow 2.5s ease-in-out infinite alternate;
}
@keyframes avataurglow {
  from { box-shadow: 0 0 12px rgba(99,102,241,.4); }
  to   { box-shadow: 0 0 24px rgba(139,92,246,.55); }
}
.chat-avatar::after {
  content: ''; position: absolute; inset: -3px; border-radius: 50%;
  border: 2px solid transparent;
  border-top-color: var(--c1); border-right-color: var(--c2);
  animation: avatarspin 3s linear infinite;
}
@keyframes avatarspin { to { transform: rotate(360deg); } }
.chat-name    { font-weight: 600; font-size: .9rem; }
.chat-status  {
  font-size: .68rem; color: var(--c4);
  display: flex; align-items: center; gap: 5px;
}
.chat-status::before {
  content: ''; width: 6px; height: 6px; border-radius: 50%;
  background: var(--c4); animation: dotpulse 1.5s infinite;
}

/* messages */
.chatbot {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.message {
  animation: msgIn .3s cubic-bezier(.34,1.56,.64,1) both !important;
}
@keyframes msgIn {
  from { opacity: 0; transform: translateY(10px) scale(.97); }
  to   { opacity: 1; transform: translateY(0)    scale(1);   }
}
/* bot — no bubble */
.message.bot .bubble-wrap,
.message.bot .content {
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 0 !important;
  font-size: .9rem !important;
  line-height: 1.75 !important;
  color: var(--tx) !important;
}
/* user — pill */
.message.user {
  display: flex !important;
  justify-content: flex-end !important;
}
.message.user .bubble-wrap,
.message.user .content {
  background: linear-gradient(135deg, rgba(99,102,241,.2), rgba(139,92,246,.15)) !important;
  border: 1px solid rgba(99,102,241,.28) !important;
  border-radius: 18px 18px 4px 18px !important;
  padding: 11px 16px !important;
  max-width: 74% !important;
  font-size: .9rem !important;
  line-height: 1.65 !important;
  color: var(--tx) !important;
}
.message p, .message span, .message li { color: var(--tx) !important; font-size: .9rem !important; }
.message strong, .message b { color: #c4b5fd !important; font-weight: 600 !important; }
.message code {
  background: rgba(99,102,241,.15) !important;
  color: #c4b5fd !important;
  border-radius: 5px !important;
  padding: 2px 6px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: .83rem !important;
}
.message em { color: var(--tx2) !important; }

/* ── THINKING ANIMATION ── */
#thinking-bar {
  display: none; align-items: center; gap: 10px;
  padding: 10px 18px 4px;
}
#thinking-bar.active { display: flex; }
#thinking-bar .t-avatar {
  width: 28px; height: 28px; border-radius: 50%;
  background: linear-gradient(135deg, var(--c1), var(--c2));
  display: flex; align-items: center; justify-content: center;
  font-size: 14px; flex-shrink: 0;
  animation: dotpulse 2s ease-in-out infinite;
}
.t-dots span {
  display: inline-block;
  width: 7px; height: 7px; border-radius: 50%;
  animation: tdot 1.3s ease-in-out infinite;
  margin-right: 5px;
}
.t-dots span:last-child { margin-right: 0; }
.t-dots span:nth-child(1) { background: var(--c1); animation-delay: 0s;   }
.t-dots span:nth-child(2) { background: var(--c2); animation-delay: .18s; }
.t-dots span:nth-child(3) { background: var(--c3); animation-delay: .36s; }
@keyframes tdot {
  0%,80%,100% { transform: translateY(0);    opacity: .4; }
  40%          { transform: translateY(-7px); opacity: 1;  }
}
.t-label { font-size: .76rem; color: var(--tx2); font-style: italic; }

/* ── QUICK REPLIES ── */
.qr-row { display: flex; gap: 8px; flex-wrap: wrap; padding: 8px 18px 4px; }
.qr-btn {
  padding: 6px 14px; border-radius: 20px;
  font-size: .73rem; font-weight: 500;
  background: rgba(99,102,241,.08);
  border: 1px solid rgba(99,102,241,.22);
  color: #a5b4fc; cursor: pointer;
  font-family: 'Space Grotesk', sans-serif;
  transition: all .2s ease;
}
.qr-btn:hover {
  background: rgba(99,102,241,.18);
  border-color: rgba(99,102,241,.45);
  color: #e0e7ff;
  transform: translateY(-1px);
}

/* ── INPUT BAR ── */
.input-bar {
  padding: 12px 16px 14px;
  border-top: 1px solid var(--br);
  display: flex; gap: 10px; align-items: flex-end;
  background: rgba(255,255,255,.015);
}
.input-bar textarea {
  border-radius: 20px !important;
  padding: 12px 18px !important;
  font-size: .9rem !important;
  min-height: 48px !important;
  background: var(--bg2) !important;
  border: 1px solid rgba(255,255,255,.1) !important;
  color: var(--tx) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  transition: border-color .2s, box-shadow .2s !important;
  resize: none !important;
}
.input-bar textarea:focus {
  border-color: rgba(99,102,241,.5) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,.1) !important;
  outline: none !important;
}
.input-bar textarea::placeholder { color: var(--tx2) !important; }

/* send button */
button.primary {
  background: linear-gradient(135deg, var(--c1), var(--c2)) !important;
  border: none !important;
  border-radius: 50% !important;
  width: 46px !important; height: 46px !important; min-width: 46px !important;
  padding: 0 !important;
  font-size: 1rem !important;
  color: #fff !important;
  box-shadow: 0 4px 20px rgba(99,102,241,.35) !important;
  transition: all .22s ease !important;
}
button.primary:hover {
  transform: scale(1.08) !important;
  box-shadow: 0 6px 28px rgba(99,102,241,.5) !important;
}
button.primary:active { transform: scale(.95) !important; }

/* secondary buttons */
button.secondary {
  background: rgba(255,255,255,.03) !important;
  border: 1px solid var(--br) !important;
  border-radius: var(--r-sm) !important;
  color: var(--tx2) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: .75rem !important;
  transition: all .2s !important;
}
button.secondary:hover {
  background: rgba(99,102,241,.1) !important;
  border-color: rgba(99,102,241,.35) !important;
  color: #c4b5fd !important;
  transform: translateY(-1px) !important;
}

/* file download */
.gr-file { background: var(--bg2) !important; border-color: var(--br) !important; }

/* ── STAR RATING BAR ── */
.star-row {
  display: flex; align-items: center; gap: 8px;
  padding: 10px 18px;
  border-top: 1px solid var(--br);
  background: rgba(255,255,255,.015);
}
.star-label {
  font-size: .72rem; color: var(--tx2);
  text-transform: uppercase; letter-spacing: .8px;
  font-weight: 600; margin-right: 4px;
}
button.star-btn {
  background: rgba(255,255,255,.03) !important;
  border: 1px solid var(--br) !important;
  border-radius: 50% !important;
  width: 36px !important; height: 36px !important; min-width: 36px !important;
  padding: 0 !important;
  font-size: 16px !important;
  cursor: pointer; transition: all .2s ease !important;
  color: #6b7280 !important;
}
button.star-btn:hover {
  background: rgba(251,191,36,.18) !important;
  border-color: rgba(251,191,36,.6) !important;
  color: #fbbf24 !important;
  transform: scale(1.15) translateY(-2px) !important;
  box-shadow: 0 6px 18px rgba(251,191,36,.25) !important;
}
button.star-btn:active { transform: scale(.95) !important; }
.star-comment textarea {
  border-radius: 14px !important;
  padding: 8px 14px !important;
  font-size: .82rem !important;
  min-height: 38px !important;
  background: var(--bg2) !important;
  border: 1px solid var(--br) !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,.3); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,.5); }
"""

    QUICK_REPLIES = [
        "🎮 Gaming laptop",
        "🎒 For college",
        "🧠 ML workstation",
        "⚖️ Compare options",
        "❓ What is GPU tier?",
        "🔍 Latest RTX news",
    ]

    LAPTOP_SVG = """
    <svg width="180" height="116" viewBox="0 0 180 116" fill="none"
         style="display:block;margin:0 auto 12px;
                filter:drop-shadow(0 8px 24px rgba(99,102,241,.35));
                animation:laptopfloat 4s ease-in-out infinite">
      <style>
        @keyframes laptopfloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
      </style>
      <!-- screen housing -->
      <rect x="15" y="4" width="150" height="88" rx="6"
            fill="#0d0f1a" stroke="rgba(99,102,241,.55)" stroke-width="1.5"/>
      <!-- screen -->
      <rect x="23" y="12" width="134" height="72" rx="3" fill="#08090f"/>
      <!-- animated content lines -->
      <rect x="29" y="19" width="122" height="7" rx="2" fill="rgba(99,102,241,.18)">
        <animate attributeName="width" values="122;86;122" dur="3.2s" repeatCount="indefinite"/>
      </rect>
      <rect x="29" y="30" width="88"  height="4" rx="2" fill="rgba(139,92,246,.14)"/>
      <rect x="29" y="38" width="104" height="4" rx="2" fill="rgba(99,102,241,.1)"/>
      <rect x="29" y="46" width="62"  height="4" rx="2" fill="rgba(16,185,129,.12)">
        <animate attributeName="width" values="62;80;62" dur="4s" repeatCount="indefinite"/>
      </rect>
      <rect x="29" y="54" width="96"  height="4" rx="2" fill="rgba(99,102,241,.09)"/>
      <rect x="29" y="62" width="74"  height="4" rx="2" fill="rgba(139,92,246,.1)"/>
      <rect x="29" y="70" width="4"   height="7" rx="1" fill="#6366f1">
        <animate attributeName="opacity" values="1;0;1" dur="1.1s" repeatCount="indefinite"/>
      </rect>
      <!-- webcam -->
      <circle cx="90" cy="8.5" r="2.5" fill="rgba(99,102,241,.35)">
        <animate attributeName="fill"
          values="rgba(99,102,241,.3);rgba(99,102,241,.8);rgba(99,102,241,.3)"
          dur="2.5s" repeatCount="indefinite"/>
      </circle>
      <!-- base -->
      <rect x="8"  y="92" width="164" height="8" rx="3"
            fill="#0d0f1a" stroke="rgba(99,102,241,.3)" stroke-width="1"/>
      <!-- trackpad -->
      <rect x="72" y="94" width="36" height="5" rx="2.5"
            fill="rgba(99,102,241,.1)" stroke="rgba(99,102,241,.2)" stroke-width=".8"/>
      <!-- bottom -->
      <path d="M3 100 Q90 110 177 100 L179 108 Q90 118 1 108 Z"
            fill="#0d0f1a" stroke="rgba(99,102,241,.16)" stroke-width="1"/>
      <!-- glow under -->
      <ellipse cx="90" cy="92" rx="52" ry="3" fill="rgba(99,102,241,.12)"/>
    </svg>
    """

    GPU_SVG = """
    <svg width="170" height="54" viewBox="0 0 170 54" fill="none"
         style="display:block;margin:0 auto 12px;
                animation:gpuglow 3s ease-in-out infinite alternate">
      <style>
        @keyframes gpuglow{
          from{filter:drop-shadow(0 3px 8px rgba(139,92,246,.25))}
          to  {filter:drop-shadow(0 3px 22px rgba(139,92,246,.6))}
        }
      </style>
      <rect x="6" y="13" width="158" height="29" rx="3"
            fill="#0d0f1a" stroke="rgba(139,92,246,.5)" stroke-width="1"/>
      <!-- fins -->
      <rect x="12" y="7"  width="6" height="19" rx="1" fill="rgba(139,92,246,.2)"  stroke="rgba(139,92,246,.32)" stroke-width=".5"/>
      <rect x="21" y="5"  width="6" height="21" rx="1" fill="rgba(139,92,246,.26)" stroke="rgba(139,92,246,.38)" stroke-width=".5"/>
      <rect x="30" y="7"  width="6" height="19" rx="1" fill="rgba(139,92,246,.2)"  stroke="rgba(139,92,246,.32)" stroke-width=".5"/>
      <rect x="39" y="4"  width="6" height="22" rx="1" fill="rgba(139,92,246,.32)" stroke="rgba(139,92,246,.44)" stroke-width=".5"/>
      <rect x="48" y="7"  width="6" height="19" rx="1" fill="rgba(139,92,246,.2)"  stroke="rgba(139,92,246,.32)" stroke-width=".5"/>
      <!-- chip -->
      <rect x="72" y="18" width="28" height="18" rx="2.5"
            fill="rgba(139,92,246,.2)" stroke="rgba(139,92,246,.55)" stroke-width=".8"/>
      <text x="86" y="30" font-family="monospace" font-size="6.5"
            fill="rgba(139,92,246,.9)" text-anchor="middle">GPU</text>
      <!-- vram -->
      <rect x="108" y="18" width="9" height="7" rx="1" fill="rgba(139,92,246,.14)" stroke="rgba(139,92,246,.28)" stroke-width=".5"/>
      <rect x="120" y="18" width="9" height="7" rx="1" fill="rgba(139,92,246,.14)" stroke="rgba(139,92,246,.28)" stroke-width=".5"/>
      <rect x="132" y="18" width="9" height="7" rx="1" fill="rgba(139,92,246,.14)" stroke="rgba(139,92,246,.28)" stroke-width=".5"/>
      <rect x="144" y="18" width="9" height="7" rx="1" fill="rgba(139,92,246,.14)" stroke="rgba(139,92,246,.28)" stroke-width=".5"/>
      <rect x="108" y="28" width="9" height="7" rx="1" fill="rgba(139,92,246,.14)" stroke="rgba(139,92,246,.28)" stroke-width=".5"/>
      <rect x="120" y="28" width="9" height="7" rx="1" fill="rgba(139,92,246,.14)" stroke="rgba(139,92,246,.28)" stroke-width=".5"/>
      <rect x="132" y="28" width="9" height="7" rx="1" fill="rgba(139,92,246,.14)" stroke="rgba(139,92,246,.28)" stroke-width=".5"/>
      <rect x="144" y="28" width="9" height="7" rx="1" fill="rgba(139,92,246,.14)" stroke="rgba(139,92,246,.28)" stroke-width=".5"/>
      <!-- pcie -->
      <rect x="8" y="39" width="154" height="5" rx="1"
            fill="rgba(139,92,246,.07)" stroke="rgba(139,92,246,.2)" stroke-width=".5"/>
      <!-- chip glow -->
      <ellipse cx="86" cy="27" rx="16" ry="8" fill="rgba(139,92,246,.07)">
        <animate attributeName="opacity" values=".5;1;.5" dur="2.2s" repeatCount="indefinite"/>
      </ellipse>
    </svg>
    """

    with gr.Blocks(title="Laptop Advisor AI", css=CSS) as demo:

        # ── TOP BAR ─────────────────────────────────────────────────
        gr.HTML(f"""
        <div id="topbar">
          <div class="topbar-logo">
            <div class="logo-orb">🖥️</div>
            <div class="logo-text">
              <h1>Laptop Advisor AI</h1>
              <p>Powered by Ollama · Local & Private · {collection.count()} laptops</p>
            </div>
          </div>
          <div class="topbar-badges">
            <span class="badge badge-indigo">⚡ Streaming</span>
            <span class="badge badge-cyan">🔍 Web Search</span>
            <span class="badge badge-emerald">🧠 Memory</span>
            <span class="badge badge-amber">📖 Wikipedia</span>
          </div>
        </div>
        """)

        # ── STAT CARDS ──────────────────────────────────────────────
        gr.HTML(f"""
        <div class="stat-row">
          <div class="stat-card">
            <div class="stat-icon">💾</div>
            <div class="stat-num">{collection.count()}</div>
            <div class="stat-label">Laptops in DB</div>
          </div>
          <div class="stat-card">
            <div class="stat-icon">⚙️</div>
            <div class="stat-num">4</div>
            <div class="stat-label">CPU Tiers</div>
          </div>
          <div class="stat-card">
            <div class="stat-icon">🎮</div>
            <div class="stat-num">4</div>
            <div class="stat-label">GPU Tiers</div>
          </div>
          <div class="stat-card">
            <div class="stat-icon">🔗</div>
            <div class="stat-num">{MAX_MEMORY}</div>
            <div class="stat-label">Memory Turns</div>
          </div>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── LEFT SIDEBAR ────────────────────────────────────────
            with gr.Column(scale=1, min_width=220):

                # Laptop illustration
                gr.HTML(f"""
                <div class="panel" style="text-align:center">
                  {LAPTOP_SVG}
                  <div style="font-size:.85rem;font-weight:700;letter-spacing:-.2px;
                       background:linear-gradient(90deg,#e0e7ff,#c4b5fd);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                       background-clip:text">AI Laptop Advisor</div>
                  <div style="font-size:.65rem;color:var(--tx2);margin-top:3px">
                    Local · Private · Offline AI
                  </div>
                </div>
                """)

                # CPU tiers
                gr.HTML("""
                <div class="panel">
                  <div class="panel-title">CPU Performance Tiers</div>
                  <div class="tier-row">
                    <div class="tier-badge" style="background:rgba(99,102,241,.12);color:#a5b4fc;border:1px solid rgba(99,102,241,.25)">1</div>
                    <div class="tier-info"><div class="tier-name" style="color:#a5b4fc">Basic</div><div class="tier-chip">i3 · Ryzen 3 · Celeron</div></div>
                  </div>
                  <div class="tier-row">
                    <div class="tier-badge" style="background:rgba(99,102,241,.2);color:#a5b4fc;border:1px solid rgba(99,102,241,.38)">2</div>
                    <div class="tier-info"><div class="tier-name" style="color:#a5b4fc">Mid-range</div><div class="tier-chip">i5 · Ryzen 5</div></div>
                  </div>
                  <div class="tier-row">
                    <div class="tier-badge" style="background:rgba(99,102,241,.32);color:#c4b5fd;border:1px solid rgba(99,102,241,.52)">3</div>
                    <div class="tier-info"><div class="tier-name" style="color:#c4b5fd">High-end</div><div class="tier-chip">i7 · Ryzen 7</div></div>
                  </div>
                  <div class="tier-row">
                    <div class="tier-badge" style="background:rgba(99,102,241,.52);color:#fff;border:1px solid rgba(99,102,241,.8)">4</div>
                    <div class="tier-info"><div class="tier-name" style="color:#e0e7ff">Top-tier</div><div class="tier-chip">i9 · Ryzen 9</div></div>
                  </div>
                </div>
                """)

                # GPU tiers
                gr.HTML(f"""
                <div class="panel">
                  <div class="panel-title">GPU Performance Tiers</div>
                  {GPU_SVG}
                  <div class="tier-row">
                    <div class="tier-badge" style="background:rgba(139,92,246,.1);color:#c4b5fd;border:1px solid rgba(139,92,246,.22)">0</div>
                    <div class="tier-info"><div class="tier-name" style="color:#c4b5fd">Integrated</div><div class="tier-chip">Intel UHD · AMD Radeon</div></div>
                  </div>
                  <div class="tier-row">
                    <div class="tier-badge" style="background:rgba(139,92,246,.18);color:#c4b5fd;border:1px solid rgba(139,92,246,.35)">1</div>
                    <div class="tier-info"><div class="tier-name" style="color:#c4b5fd">Entry Gaming</div><div class="tier-chip">GTX 1650 · RTX 3050</div></div>
                  </div>
                  <div class="tier-row">
                    <div class="tier-badge" style="background:rgba(139,92,246,.3);color:#ddd6fe;border:1px solid rgba(139,92,246,.5)">2</div>
                    <div class="tier-info"><div class="tier-name" style="color:#ddd6fe">Mid Gaming</div><div class="tier-chip">RTX 3060 · RTX 4060</div></div>
                  </div>
                  <div class="tier-row">
                    <div class="tier-badge" style="background:rgba(139,92,246,.52);color:#fff;border:1px solid rgba(139,92,246,.8)">3</div>
                    <div class="tier-info"><div class="tier-name" style="color:#ede9fe">High-end</div><div class="tier-chip">RTX 3080 · RTX 4080+</div></div>
                  </div>
                </div>
                """)

                # Capabilities
                gr.HTML("""
                <div class="panel">
                  <div class="panel-title">Capabilities</div>
                  <div class="cap-row">🖥️ 1020 laptop database<span class="cap-dot"></span></div>
                  <div class="cap-row">🔍 DuckDuckGo search<span class="cap-dot" style="animation-delay:.4s"></span></div>
                  <div class="cap-row">📖 Wikipedia lookups<span class="cap-dot" style="animation-delay:.8s"></span></div>
                  <div class="cap-row">🔗 Read any URL<span class="cap-dot" style="animation-delay:1.2s"></span></div>
                  <div class="cap-row">🧠 20-turn memory<span class="cap-dot" style="animation-delay:1.6s"></span></div>
                  <div class="cap-row">⚡ Real-time streaming<span class="cap-dot" style="animation-delay:2s"></span></div>
                </div>
                """)

                # ── LIVE FEEDBACK DASHBOARD ──
                dashboard_html = gr.HTML(render_dashboard())

                reset_btn = gr.Button("🔄 Start Over", variant="secondary")

            # ── CHAT AREA ───────────────────────────────────────────
            with gr.Column(scale=4):
                gr.HTML("""
                <div class="chat-wrap" style="border-radius:14px;overflow:hidden">

                  <!-- chat header -->
                  <div class="chat-header">
                    <div class="chat-avatar">🤖</div>
                    <div>
                      <div class="chat-name">Laptop Advisor</div>
                      <div class="chat-status">Online · Ready</div>
                    </div>
                  </div>

                  <!-- thinking animation (shown/hidden by JS) -->
                  <div id="thinking-bar">
                    <div class="t-avatar">🤖</div>
                    <div class="t-dots">
                      <span></span><span></span><span></span>
                    </div>
                    <span class="t-label">thinking...</span>
                  </div>

                </div>

                <style>
                  /* wire thinking to Gradio chatbot below */
                  #thinking-bar { display:none; align-items:center; gap:10px; padding:10px 18px 4px; }
                  #thinking-bar.active { display:flex; }
                </style>
                <script>
                (function(){
                  function init(){
                    const bar     = document.getElementById('thinking-bar');
                    const chatbox = document.querySelector('.chatbot');
                    const sendBtn = document.querySelector('button.primary');
                    if(!bar||!chatbox||!sendBtn){ setTimeout(init,400); return; }
                    const show = ()=> bar.classList.add('active');
                    const hide = ()=> bar.classList.remove('active');
                    sendBtn.addEventListener('click', show);
                    document.addEventListener('keydown', e=>{
                      if(e.key==='Enter'&&!e.shiftKey) show();
                    });
                    new MutationObserver(hide).observe(chatbox,{childList:true,subtree:true});
                  }
                  init();
                })();
                </script>
                """)

                chatbot = gr.Chatbot(
                    value=[to_msg("assistant", FIRST_MSG)],
                    label="", height=500, show_label=False,
                    elem_classes=["chatbot"],
                )

                # Quick replies
                gr.HTML("""<div class="qr-row" id="qr-row"></div>""")

                with gr.Row():
                    qr_btns = []
                    for label in QUICK_REPLIES:
                        btn = gr.Button(label, size="sm", variant="secondary")
                        qr_btns.append((btn, label))

                # Input bar
                with gr.Row(elem_classes=["input-bar"]):
                    msg_box = gr.Textbox(
                        placeholder="Message Laptop Advisor... (1=guided, 2=free, or ask anything)",
                        show_label=False, scale=6, container=False,
                    )
                    send_btn = gr.Button("➤", variant="primary", scale=1)

                with gr.Row():
                    export_btn  = gr.Button("💾 Export Chat", variant="secondary", scale=1)
                    export_file = gr.File(label="Download", visible=False, scale=3)

                # ── STAR RATING BAR ──
                gr.HTML('<div style="font-size:.7rem;color:#9499c4;padding:10px 4px 4px;letter-spacing:.5px;text-transform:uppercase;font-weight:600">Rate this response</div>')
                with gr.Row():
                    star1 = gr.Button("★", variant="secondary", scale=0, elem_classes=["star-btn"])
                    star2 = gr.Button("★", variant="secondary", scale=0, elem_classes=["star-btn"])
                    star3 = gr.Button("★", variant="secondary", scale=0, elem_classes=["star-btn"])
                    star4 = gr.Button("★", variant="secondary", scale=0, elem_classes=["star-btn"])
                    star5 = gr.Button("★", variant="secondary", scale=0, elem_classes=["star-btn"])
                    fb_comment = gr.Textbox(
                        placeholder="Click 1-5 stars · Add a comment (optional) and press Enter",
                        show_label=False, scale=5, container=False,
                        elem_classes=["star-comment"],
                    )

        # ── EVENT WIRING — all real Gradio, nothing changed ──
        send_btn.click(chat, [msg_box, chatbot], [msg_box, chatbot])
        msg_box.submit(chat, [msg_box, chatbot], [msg_box, chatbot])
        reset_btn.click(reset, outputs=[chatbot, msg_box])
        export_btn.click(export, inputs=[chatbot], outputs=[export_file])

        # Star rating wiring — each star submits its number with the current comment
        star1.click(lambda c,h: submit_star(1, c, h), inputs=[fb_comment, chatbot], outputs=[dashboard_html, fb_comment])
        star2.click(lambda c,h: submit_star(2, c, h), inputs=[fb_comment, chatbot], outputs=[dashboard_html, fb_comment])
        star3.click(lambda c,h: submit_star(3, c, h), inputs=[fb_comment, chatbot], outputs=[dashboard_html, fb_comment])
        star4.click(lambda c,h: submit_star(4, c, h), inputs=[fb_comment, chatbot], outputs=[dashboard_html, fb_comment])
        star5.click(lambda c,h: submit_star(5, c, h), inputs=[fb_comment, chatbot], outputs=[dashboard_html, fb_comment])

        for btn, label in qr_btns:
            btn.click(fn=lambda l=label: l, inputs=[], outputs=[msg_box])

    return demo

if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv or not os.path.exists("./chroma_db")
    if rebuild:
        print("[Setup] Building vector DB from CSV...")
        collection = build_vectorstore(load_data(CSV_PATH))
    else:
        print("[Setup] Loading existing vector DB...")
        try:
            collection = load_vectorstore()
        except Exception:
            print("[Setup] Not found — rebuilding...")
            collection = build_vectorstore(load_data(CSV_PATH))

    print(f"\n[Ollama] Model: {OLLAMA_MODEL}")
    print("[UI] Launching on http://localhost:7860\n")
    demo = build_ui(collection)
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)