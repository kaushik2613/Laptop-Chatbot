"""
Laptop RAG Chatbot — powered by Ollama + ChromaDB
Dataset: final1_tiered.csv (1020 laptops with CPU/GPU tier scores)
Compatible with Gradio 6.11+
"""

import os
import sys
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import ollama
import gradio as gr

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CSV_PATH     = "final1_tiered.csv"
OLLAMA_MODEL = "llama3.2"
COLLECTION   = "laptops_v3"
TOP_K        = 5
EMBED_MODEL  = "all-MiniLM-L6-v2"

GUIDED_QUESTIONS = [
    "What will you mainly use this laptop for? (e.g. gaming, work, school, video editing, coding)",
    "What is your budget in INR? (e.g. under ₹40,000 / ₹40k-₹80k / ₹80k-₹1.5L / no limit)",
    "Do you prefer any brand? (Dell, HP, Lenovo, Apple, ASUS, MSI, or no preference)",
    "How important is portability? (lightweight & thin / don't care / prefer larger screen)",
    "Do you need a dedicated GPU for gaming or ML workloads? (yes / no)",
]

# ─────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────
def build_spec_string(row):
    """
    Builds a rich text string from all structured columns.
    Includes tier scores so the embedding carries performance context.
    """
    parts = []

    parts.append(f"Brand: {row['Brand']}")
    parts.append(f"Processor: {row.get('Processor_full', 'N/A')}")

    # CPU tier — 1=basic 2=mid 3=high 4=top
    cpu_tier = row.get('cpu_tier', 1)
    cpu_tier_label = {1:"Basic", 2:"Mid-range", 3:"High-end", 4:"Top-tier"}.get(int(cpu_tier), "Mid-range")
    parts.append(f"CPU Tier: {int(cpu_tier)} ({cpu_tier_label})")
    parts.append(f"CPU Single-core Score: {row.get('b_singleScore', 'N/A')}")
    parts.append(f"CPU Multi-core Score: {row.get('b_multiScore', 'N/A')}")

    parts.append(f"RAM: {row['RAM_GB']}GB {row.get('RAM_type', '')}")
    parts.append(f"Storage: {row['Storage_capacity_GB']}GB {row.get('Storage_type', '')}")

    # GPU
    gpu_name = str(row.get('Graphics_name', 'N/A'))
    gpu_tier = row.get('gpu_tier', 0)
    gpu_tier_label = {0:"Integrated", 1:"Entry", 2:"Mid-range", 3:"High-end"}.get(int(gpu_tier), "Integrated")
    integrated = row.get('Graphics_integreted', True)
    gpu_type = "Integrated" if integrated else "Dedicated"
    gpu_gb = row.get('Graphics_GB', '')
    gpu_mem = f" {gpu_gb}GB" if pd.notna(gpu_gb) and gpu_gb else ""
    parts.append(f"GPU: {gpu_name}{gpu_mem} ({gpu_type})")
    parts.append(f"GPU Tier: {int(gpu_tier)} ({gpu_tier_label})")
    parts.append(f"GPU 3D Score: {row.get('b_G3Dmark', 'N/A')}")

    # Display
    size  = row.get('Display_size_inches', 'N/A')
    ppi   = row.get('ppi', 0)
    h_px  = row.get('Horizontal_pixel', '')
    v_px  = row.get('Vertical_pixel', '')
    touch = "Touchscreen" if row.get('Touch_screen', False) else "Non-touch"
    parts.append(f"Display: {size} inch {h_px}x{v_px} {float(ppi):.0f}ppi {touch}")

    # Price
    price = row.get('Price', 0)
    parts.append(f"Price: Rs{int(price):,}")

    parts.append(f"OS: {row.get('Operating_system', 'N/A')}")

    rating = row.get('Rating', '')
    if pd.notna(rating) and rating:
        parts.append(f"Rating: {rating}/5")

    return " | ".join(parts)


def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")

    # Build spec string from all columns including tiers
    df['specs'] = df.apply(build_spec_string, axis=1)
    df['model'] = df['Name'].astype(str).str.strip()
    df['price'] = df['Price'] if 'Price' in df.columns else 0

    print(f"[Data] Loaded {len(df)} laptops from {path}")
    print(f"[Data] Sample spec:\n       {df['specs'].iloc[0][:200]}...")
    return df

# ─────────────────────────────────────────────
# 2. VECTOR STORE
# ─────────────────────────────────────────────
def build_vectorstore(df):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection(COLLECTION)
    except:
        pass
    col = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    docs, ids, metas = [], [], []
    for i, row in df.iterrows():
        docs.append(f"Model: {row['model']}\nSpecs: {row['specs']}")
        ids.append(str(i))
        metas.append({
            "model":     row["model"],
            "brand":     str(row.get("Brand", "")),
            "price":     str(int(row.get("price", 0))),
            "cpu_tier":  str(int(row.get("cpu_tier", 1))),
            "gpu_tier":  str(int(row.get("gpu_tier", 0))),
            "link":      ""
        })

    for start in range(0, len(docs), 500):
        col.add(
            documents=docs[start:start+500],
            ids=ids[start:start+500],
            metadatas=metas[start:start+500]
        )
        print(f"[VectorDB] Indexed {min(start+500, len(docs))}/{len(docs)}")

    print(f"[VectorDB] Ready — {col.count()} laptops indexed")
    return col


def load_vectorstore():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path="./chroma_db")
    col = client.get_collection(name=COLLECTION, embedding_function=ef)
    print(f"[VectorDB] Loaded — {col.count()} laptops")
    return col

# ─────────────────────────────────────────────
# 3. RETRIEVAL
# score = 1 - cosine_distance  (0 = no match, 1 = perfect)
# ─────────────────────────────────────────────
def retrieve(collection, query, k=TOP_K):
    res = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({
            "document":  doc,
            "model":     meta.get("model", ""),
            "brand":     meta.get("brand", ""),
            "price":     meta.get("price", "0"),
            "cpu_tier":  meta.get("cpu_tier", "1"),
            "gpu_tier":  meta.get("gpu_tier", "0"),
            "score":     round(1 - dist, 3),
        })
    return hits

# ─────────────────────────────────────────────
# 4. LLM
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert laptop advisor for the Indian market.
Help users find the best laptop using real product data. All prices are in Indian Rupees (INR, ₹).

Rules:
- Only recommend laptops present in the provided context.
- Always mention the price in ₹ (INR).
- Explain WHY each laptop fits the user's needs based on specs and tier scores.
- CPU Tier: 1=Basic, 2=Mid-range, 3=High-end, 4=Top-tier.
- GPU Tier: 0=Integrated, 1=Entry gaming, 2=Mid gaming, 3=High-end gaming.
- Mention CPU tier, GPU tier, RAM, Storage, Display size, and Price for each recommendation.
- If the user asks for a budget in USD, convert: 1 USD ≈ ₹83.
- If no good match exists, say so honestly.
- Be concise but specific. Format recommendations as a numbered list."""


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return " ".join(parts)
    return str(content)


def generate_answer(query, hits, history):
    context = ""
    for i, h in enumerate(hits, 1):
        context += (
            f"\n--- Laptop {i} "
            f"(similarity: {h['score']} | Price: ₹{h['price']} | "
            f"CPU Tier: {h['cpu_tier']} | GPU Tier: {h['gpu_tier']}) ---\n"
            f"{h['document']}\n"
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-12:]:
        if isinstance(turn, dict):
            messages.append({
                "role":    turn["role"],
                "content": extract_text(turn["content"])
            })
    messages.append({"role": "user", "content":
        f"User question: {query}\n\n"
        f"Relevant laptops from database:\n{context}\n\n"
        f"Give a clear recommendation based only on the laptops above."
    })

    try:
        resp = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        return resp["message"]["content"]
    except Exception as e:
        return (
            f"Ollama error: {e}\n\n"
            f"Make sure Ollama is running: ollama pull {OLLAMA_MODEL}"
        )

# ─────────────────────────────────────────────
# 5. ONBOARDING STATE
# ─────────────────────────────────────────────
class OnboardingState:
    def __init__(self):
        self.answers = []
        self.q_index = 0
        self.done    = False

    def current_question(self):
        return GUIDED_QUESTIONS[self.q_index]

    def answer(self, text):
        self.answers.append(text)
        self.q_index += 1
        if self.q_index >= len(GUIDED_QUESTIONS):
            self.done = True

    def as_query(self):
        return " ".join(self.answers)

    def summary(self):
        return "\n".join(
            f"• {q}\n  → {a}"
            for q, a in zip(GUIDED_QUESTIONS, self.answers)
        )


def to_msg(role, content):
    return {"role": role, "content": content}

# ─────────────────────────────────────────────
# 6. UI
# ─────────────────────────────────────────────
def build_ui(collection):
    state = {"ob": OnboardingState()}

    FIRST_MSG = (
        f"👋 Hi! I'm your laptop advisor — powered by a local AI.\n\n"
        f"I have **{collection.count()} laptops** in my database with CPU & GPU performance tiers.\n"
        f"All prices are in ₹ INR.\n\n"
        f"I'll ask you {len(GUIDED_QUESTIONS)} quick questions to find your perfect match.\n\n"
        f"**Question 1/{len(GUIDED_QUESTIONS)}:** {GUIDED_QUESTIONS[0]}"
    )

    def chat(user_msg, history):
        if not user_msg.strip():
            return "", history

        ob = state["ob"]

        if not ob.done:
            ob.answer(user_msg)
            if not ob.done:
                bot = (
                    f"**Question {ob.q_index + 1}/{len(GUIDED_QUESTIONS)}:** "
                    f"{ob.current_question()}"
                )
            else:
                hits   = retrieve(collection, ob.as_query())
                answer = generate_answer(
                    f"Recommend laptops based on these preferences:\n{ob.summary()}",
                    hits, []
                )
                bot = (
                    f"Got your preferences! Here's what I found:\n\n{answer}\n\n"
                    f"---\nFeel free to ask follow-up questions like:\n"
                    f"- Show me options under ₹50,000\n"
                    f"- Which has the best GPU for gaming?\n"
                    f"- Any lightweight options under ₹60,000?"
                )
        else:
            hits   = retrieve(collection, user_msg)
            answer = generate_answer(user_msg, hits, history)
            sources = "\n".join(
                f"  • {h['model']} — ₹{h['price']} "
                f"| CPU T{h['cpu_tier']} GPU T{h['gpu_tier']} "
                f"(score: {h['score']})"
                for h in hits if h["model"]
            )
            bot = f"{answer}\n\n📚 Sources checked:\n{sources}"

        history = history + [to_msg("user", user_msg), to_msg("assistant", bot)]
        return "", history

    def reset():
        state["ob"] = OnboardingState()
        return [to_msg("assistant", FIRST_MSG)], ""

    with gr.Blocks(title="Laptop Advisor") as demo:
        gr.HTML("""
        <div style="text-align:center;padding:20px 0 10px">
            <h1 style="font-size:1.8rem;margin:0">🖥️ Laptop Advisor</h1>
            <p style="color:#888;margin:6px 0 0;font-size:0.9rem">
                RAG-powered &middot; Local Ollama &middot; 1020 laptops &middot;
                CPU/GPU Tiers &middot; Prices in ₹ INR
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[to_msg("assistant", FIRST_MSG)],
                    label="Chat",
                    height=520,
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Type your answer or question...",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")
                gr.Markdown(f"**LLM:** `{OLLAMA_MODEL}`")
                gr.Markdown(f"**Embeddings:** `{EMBED_MODEL}`")
                gr.Markdown(f"**Top-K:** `{TOP_K}`")
                gr.Markdown(f"**Database:** `{collection.count()} laptops`")
                gr.Markdown("**Currency:** `₹ INR`")
                reset_btn = gr.Button("🔄 Start Over", variant="secondary")

                gr.Markdown("### 🎯 CPU Tiers")
                gr.Markdown("1 = Basic (i3/Ryzen 3)\n\n2 = Mid (i5/Ryzen 5)\n\n3 = High (i7/Ryzen 7)\n\n4 = Top (i9/Ryzen 9)")

                gr.Markdown("### 🎮 GPU Tiers")
                gr.Markdown("0 = Integrated\n\n1 = Entry gaming\n\n2 = Mid gaming\n\n3 = High-end")

                gr.Markdown("### 💡 Example Questions")
                gr.Examples(
                    examples=[
                        ["Best gaming laptop under ₹80,000"],
                        ["Lightweight laptop for college under ₹50,000"],
                        ["Best laptop for video editing with high CPU tier"],
                        ["Budget laptop under ₹30,000"],
                        ["Best laptop with GPU tier 3 for ML work"],
                        ["MacBook alternatives with good battery life"],
                    ],
                    inputs=msg_box,
                )

        send_btn.click(chat, [msg_box, chatbot], [msg_box, chatbot])
        msg_box.submit(chat, [msg_box, chatbot], [msg_box, chatbot])
        reset_btn.click(reset, outputs=[chatbot, msg_box])

    return demo

# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
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
            print("[Setup] Not found — rebuilding from CSV...")
            collection = build_vectorstore(load_data(CSV_PATH))

    print(f"\n[Ollama] Model: {OLLAMA_MODEL}")
    print("[UI] Launching on http://localhost:7860\n")

    demo = build_ui(collection)
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)