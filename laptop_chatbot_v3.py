"""
Laptop RAG Chatbot v11 — Full Custom HTML Dashboard
The entire UI is a custom HTML page served by Gradio,
with animated SVG laptop/GPU/CPU illustrations, particle
background, animated chatbot avatar, and modern card layouts.
"""

import os, sys, re, json
from datetime import datetime
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import ollama
import gradio as gr

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

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
# WEB TOOLS
# ─────────────────────────────────────────────
def web_search(query, max_results=4):
    if not WEB_AVAILABLE: return []
    try:
        h = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}", headers=h, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        out=[]
        for res in soup.find_all("div", class_="result", limit=max_results):
            t=res.find("a",class_="result__a"); s=res.find("a",class_="result__snippet")
            if t: out.append({"title":t.get_text(strip=True),"url":t.get("href",""),"snippet":s.get_text(strip=True) if s else ""})
        return out
    except: return []

def fetch_page(url, max_chars=3000):
    if not WEB_AVAILABLE: return ""
    try:
        r=requests.get(url,headers={"User-Agent":"Mozilla/5.0"},timeout=10)
        soup=BeautifulSoup(r.text,"html.parser")
        for t in soup(["script","style","nav","footer","header"]): t.decompose()
        return re.sub(r"\s+"," ",soup.get_text(separator=" ",strip=True))[:max_chars]
    except: return ""

def wikipedia_summary(topic):
    if not WEB_AVAILABLE: return ""
    try:
        r=requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ','_')}",
                       headers={"User-Agent":"LaptopAdvisor/1.0"},timeout=8)
        if r.status_code==200: return r.json().get("extract","")[:1500]
    except: pass
    return ""

def fmt_results(results):
    if not results: return "No web results found."
    return "\n\n".join(f"[{i}] {r['title']}\n{r['snippet']}\nURL: {r['url']}" for i,r in enumerate(results,1))

def route_query(msg, mode):
    m=msg.lower().strip()
    if re.search(r"https?://\S+",msg): return "url_fetch"
    if any(x in m for x in ["compare","vs","versus","option 1","option 2"]): return "comparison"
    if any(x in m for x in ["what is","what are","explain","how does","define"]):
        if any(t in m for t in ["cpu","gpu","ram","ssd","oled","ips","ghz","vram","ddr","rtx","nvme"]): return "wikipedia"
        return "web_search"
    if any(x in m for x in ["latest","2024","2025","review","benchmark","price","buy","news"]): return "web_search"
    if any(x in m for x in ["enough","is it","good for","can it","better for","lighter","heavier","battery","alternative"]) and mode=="done": return "followup"
    return "laptop_db"

# ─────────────────────────────────────────────
# GUIDED Q&A
# ─────────────────────────────────────────────
GUIDED_QUESTIONS = [
    "What are you mainly going to use this laptop for?\n\n  1️⃣  Gaming\n  2️⃣  Video editing / creative work\n  3️⃣  Coding / development\n  4️⃣  School / everyday tasks\n  5️⃣  Machine learning / AI",
    "How much RAM do you need?\n\n  1️⃣  8GB — light use\n  2️⃣  16GB — sweet spot\n  3️⃣  32GB — heavy workloads\n  4️⃣  No preference",
    "What screen size?\n\n  1️⃣  13–14 inch — portable\n  2️⃣  15–16 inch — balanced\n  3️⃣  17+ inch — large screen\n  4️⃣  No preference",
    "Do you need a dedicated GPU?\n\n  1️⃣  Yes — gaming / editing / ML\n  2️⃣  No — integrated is fine\n  3️⃣  Not sure",
    "Brand preference?\n\n  1️⃣  HP   2️⃣  Dell   3️⃣  Lenovo\n  4️⃣  ASUS   5️⃣  Apple   6️⃣  MSI/Gigabyte/Razer\n  7️⃣  No preference",
]
ACKS=[
    {"1":"Nice! Gaming laptops need serious hardware. Strong GPU and CPU coming up!","2":"Creative work needs good CPU and display. Got it!","3":"Coder's machine — solid CPU and plenty of RAM!","4":"Light and reliable — makes sense!","5":"ML needs top-tier CPU and dedicated GPU. On it!","default":"Got it!"},
    {"1":"8GB for light use.","2":"16GB — the sweet spot!","3":"32GB for serious workloads.","4":"I'll match to your use case.","default":"Noted."},
    {"1":"Compact and portable!","2":"15–16 inch — most popular size.","3":"Big screen for work or gaming!","4":"I'll pick the right size.","default":"Perfect."},
    {"1":"Dedicated GPU it is!","2":"Integrated works great for your needs.","3":"I'll figure it out.","default":"Got it."},
    {"1":"HP!","2":"Dell!","3":"Lenovo!","4":"ASUS!","5":"MacBook!","6":"Gaming brand!","7":"No preference — I'll find the absolute best!","default":"Noted!"},
]
QMAPS=[
    {"1":"gaming high performance dedicated GPU RTX CPU tier 3","2":"video editing creative strong CPU GPU","3":"coding development fast CPU 16GB RAM","4":"school college everyday lightweight portable","5":"machine learning AI top CPU GPU 32GB"},
    {"1":"8GB RAM","2":"16GB RAM","3":"32GB RAM","4":""},
    {"1":"13 inch 14 inch portable","2":"15 inch 16 inch","3":"17 inch large screen","4":""},
    {"1":"dedicated GPU gaming tier 2 tier 3","2":"integrated GPU","3":""},
    {"1":"HP","2":"Dell","3":"Lenovo","4":"ASUS","5":"Apple MacBook","6":"MSI Gigabyte Razer gaming","7":""},
]
def get_ack(qi,ans): return ACKS[qi].get(ans.strip(),ACKS[qi].get("default","Got it!"))
def expand(qi,ans):
    a=ans.strip(); m=QMAPS[qi]
    return m.get(a,a) if m.get(a) else a

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def build_spec_string(row):
    p=[]
    p.append(f"Brand: {row['Brand']}")
    p.append(f"Processor: {row.get('Processor_full','N/A')}")
    ct=int(row.get('cpu_tier',1)); p.append(f"CPU Tier: {ct} ({['','Basic','Mid-range','High-end','Top-tier'][ct]})")
    p.append(f"CPU Single: {row.get('b_singleScore','N/A')}"); p.append(f"CPU Multi: {row.get('b_multiScore','N/A')}")
    p.append(f"RAM: {row['RAM_GB']}GB {row.get('RAM_type','')}"); p.append(f"Storage: {row['Storage_capacity_GB']}GB {row.get('Storage_type','')}")
    gt=int(row.get('gpu_tier',0)); gn=str(row.get('Graphics_name','N/A'))
    ig=row.get('Graphics_integreted',True); gb=row.get('Graphics_GB',''); gm=f" {gb}GB" if pd.notna(gb) and gb else ""
    p.append(f"GPU: {gn}{gm} ({'Integrated' if ig else 'Dedicated'})")
    p.append(f"GPU Tier: {gt} ({['Integrated','Entry','Mid-range','High-end'][gt]})")
    p.append(f"GPU 3D: {row.get('b_G3Dmark','N/A')}")
    sz=row.get('Display_size_inches','N/A'); pp=row.get('ppi',0); tc="Touchscreen" if row.get('Touch_screen',False) else "Non-touch"
    p.append(f"Display: {sz}in {float(pp):.0f}ppi {tc}"); p.append(f"OS: {row.get('Operating_system','N/A')}")
    return " | ".join(p)

def load_data(path):
    df=pd.read_csv(path)
    df=df.drop(columns=[c for c in df.columns if "Unnamed" in c],errors="ignore")
    df=df.dropna(subset=["Name"]); df['specs']=df.apply(build_spec_string,axis=1); df['model']=df['Name'].astype(str).str.strip()
    print(f"[Data] Loaded {len(df)} laptops"); return df

def build_vectorstore(df):
    ef=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client=chromadb.PersistentClient(path="./chroma_db")
    try: client.delete_collection(COLLECTION)
    except: pass
    col=client.create_collection(name=COLLECTION,embedding_function=ef,metadata={"hnsw:space":"cosine"})
    docs,ids,metas=[],[],[]
    for i,row in df.iterrows():
        docs.append(f"Model: {row['model']}\nSpecs: {row['specs']}"); ids.append(str(i))
        metas.append({"model":row["model"],"brand":str(row.get("Brand","")),"ram":str(row.get("RAM_GB","")),"storage":str(row.get("Storage_capacity_GB","")),"display":str(row.get("Display_size_inches","")),"cpu_tier":str(int(row.get("cpu_tier",1))),"gpu_tier":str(int(row.get("gpu_tier",0))),"os":str(row.get("Operating_system",""))})
    for s in range(0,len(docs),500):
        col.add(documents=docs[s:s+500],ids=ids[s:s+500],metadatas=metas[s:s+500])
        print(f"[VectorDB] Indexed {min(s+500,len(docs))}/{len(docs)}")
    print(f"[VectorDB] Ready — {col.count()}"); return col

def load_vectorstore():
    ef=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client=chromadb.PersistentClient(path="./chroma_db")
    col=client.get_collection(name=COLLECTION,embedding_function=ef)
    print(f"[VectorDB] Loaded — {col.count()}"); return col

def retrieve(col,query,k=TOP_K):
    res=col.query(query_texts=[query],n_results=k,include=["documents","metadatas","distances"])
    hits=[]
    for doc,meta,dist in zip(res["documents"][0],res["metadatas"][0],res["distances"][0]):
        hits.append({"document":doc,"model":meta.get("model",""),"brand":meta.get("brand",""),"ram":meta.get("ram",""),"storage":meta.get("storage",""),"display":meta.get("display",""),"cpu_tier":meta.get("cpu_tier","1"),"gpu_tier":meta.get("gpu_tier","0"),"os":meta.get("os",""),"score":round(1-dist,3)})
    return hits

def build_ctx(hits):
    ctx=""
    for i,h in enumerate(hits,1):
        ctx+=f"\n--- Laptop {i} ---\nScore:{h['score']} RAM:{h['ram']}GB Storage:{h['storage']}GB Display:{h['display']}in CPU_Tier:{h['cpu_tier']} GPU_Tier:{h['gpu_tier']} OS:{h['os']}\n{h['document']}\n"
    return ctx

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
SYS="""You are a friendly, knowledgeable laptop advisor AND general tech assistant.
Be warm and conversational. Explain tech terms simply.
For laptop recommendations always give 3 options with a "My Pick" at the end.
CPU Tier: 1=Basic(i3/Ryzen3), 2=Mid(i5/Ryzen5), 3=High(i7/Ryzen7), 4=Top(i9/Ryzen9)
GPU Tier: 0=Integrated, 1=Entry, 2=Mid(RTX3060), 3=High(RTX3070+)"""

REC_SYS="""You are a friendly laptop advisor. Recommend exactly 3 laptops.
Format each as:
Option [N] — [Name]
[Why it fits]
Specs:
- Processor: [name] (CPU Tier [X]/4) — [plain English]
- Graphics: [GPU] (GPU Tier [X]/3) — [capability]
- Memory: [X]GB — [adequate/ideal/overkill]
- Storage: [X]GB [type]
- Screen: [X] inch — [portability comment]
- OS: [OS]
Worth knowing: [trade-off]
---
My Pick: [2-3 casual sentences]"""

def xt(c):
    if isinstance(c,str): return c
    if isinstance(c,list): return " ".join(p if isinstance(p,str) else p.get("text","") for p in c)
    return str(c)

def bmsgs(sys,hist,uc):
    m=[{"role":"system","content":sys}]
    for t in hist[-MAX_MEMORY:]:
        if isinstance(t,dict): m.append({"role":t["role"],"content":xt(t["content"])})
    m.append({"role":"user","content":uc}); return m

def stream_llm(messages):
    try:
        full=""
        for chunk in ollama.chat(model=OLLAMA_MODEL,messages=messages,stream=True):
            tok=chunk.get("message",{}).get("content","")
            if tok: full+=tok; yield full
        if not full: yield "I couldn't generate a response. Please try again."
    except Exception as e:
        yield f"Error: {e}\n\nMake sure Ollama is running: ollama pull {OLLAMA_MODEL}"

def export_chat(history):
    if not history: return None
    ts=datetime.now().strftime("%Y%m%d_%H%M%S"); fp=os.path.join(os.getcwd(),f"laptop_chat_{ts}.txt")
    with open(fp,"w",encoding="utf-8") as f:
        f.write(f"Laptop Advisor Chat — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'='*60}\n\n")
        for t in history:
            if isinstance(t,dict): f.write(f"{'You' if t['role']=='user' else 'Advisor'}:\n{xt(t['content'])}\n\n{'─'*40}\n\n")
    return fp

# ─────────────────────────────────────────────
# ONBOARDING
# ─────────────────────────────────────────────
class Onboard:
    def __init__(self): self.ans=[]; self.exp=[]; self.qi=0; self.done=False
    def q(self): return GUIDED_QUESTIONS[self.qi]
    def answer(self,t):
        self.ans.append(t); self.exp.append(expand(self.qi,t)); self.qi+=1
        if self.qi>=len(GUIDED_QUESTIONS): self.done=True
    def ack(self): return get_ack(self.qi-1,self.ans[-1])
    def query(self): return " ".join(a for a in self.exp if a)
    def summary(self):
        lb=["Use case","RAM","Screen","GPU","Brand"]
        return "\n".join(f"  {lb[i]}: {a}" for i,a in enumerate(self.ans))

def tm(r,c): return {"role":r,"content":c}

# ─────────────────────────────────────────────
# CUSTOM HTML — full dashboard
# ─────────────────────────────────────────────
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Laptop Advisor AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg0: #04080f;
  --bg1: #080d18;
  --bg2: #0d1525;
  --bg3: #111c30;
  --accent: #0af;
  --accent2: #a855f7;
  --accent3: #f97316;
  --green: #22d3a0;
  --text: #e2eaf8;
  --muted: #64748b;
  --border: rgba(255,255,255,0.07);
  --glass: rgba(255,255,255,0.04);
  --r: 16px;
}
*{margin:0;padding:0;box-sizing:border-box}
body{
  font-family:'Outfit',sans-serif;
  background:var(--bg0);
  color:var(--text);
  min-height:100vh;
  overflow-x:hidden;
}

/* ── CANVAS BACKGROUND ── */
#bg-canvas{
  position:fixed;inset:0;z-index:0;
  pointer-events:none;
}

/* ── GRID OVERLAY ── */
.grid-overlay{
  position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:
    linear-gradient(rgba(0,170,255,0.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,170,255,0.03) 1px,transparent 1px);
  background-size:48px 48px;
  mask-image:radial-gradient(ellipse 80% 80% at 50% 50%,#000 40%,transparent 100%);
}

/* ── LAYOUT ── */
.app{
  position:relative;z-index:1;
  display:grid;
  grid-template-columns:1fr 380px;
  grid-template-rows:auto 1fr;
  min-height:100vh;
  max-width:1440px;
  margin:0 auto;
  padding:24px;
  gap:20px;
}

/* ── HEADER ── */
.header{
  grid-column:1/-1;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 4px 8px;
  border-bottom:1px solid var(--border);
}
.logo{
  display:flex;align-items:center;gap:14px;
}
.logo-icon{
  width:46px;height:46px;border-radius:12px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;
  font-size:22px;
  box-shadow:0 0 24px rgba(0,170,255,0.3);
  animation:iconPulse 3s ease-in-out infinite;
}
@keyframes iconPulse{
  0%,100%{box-shadow:0 0 24px rgba(0,170,255,0.3);}
  50%{box-shadow:0 0 40px rgba(0,170,255,0.5),0 0 60px rgba(168,85,247,0.2);}
}
.logo-text h1{
  font-size:1.5rem;font-weight:800;letter-spacing:-0.5px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.logo-text p{font-size:0.75rem;color:var(--muted);font-weight:400;margin-top:1px}

.header-badges{display:flex;gap:8px;flex-wrap:wrap;}
.badge{
  padding:5px 12px;border-radius:20px;font-size:0.72rem;font-weight:600;
  letter-spacing:0.5px;border:1px solid;
}
.badge-cyan{background:rgba(0,170,255,0.1);border-color:rgba(0,170,255,0.3);color:var(--accent);}
.badge-purple{background:rgba(168,85,247,0.1);border-color:rgba(168,85,247,0.3);color:var(--accent2);}
.badge-green{background:rgba(34,211,160,0.1);border-color:rgba(34,211,160,0.3);color:var(--green);}
.badge-orange{background:rgba(249,115,22,0.1);border-color:rgba(249,115,22,0.3);color:var(--accent3);}

/* ── MAIN CHAT PANEL ── */
.chat-panel{
  display:flex;flex-direction:column;gap:16px;
  min-height:0;
}

/* ── HERO STATS ── */
.stats-row{
  display:grid;grid-template-columns:repeat(4,1fr);gap:12px;
}
.stat-card{
  background:var(--glass);
  border:1px solid var(--border);
  border-radius:var(--r);
  padding:16px;
  backdrop-filter:blur(20px);
  position:relative;overflow:hidden;
  transition:transform 0.25s ease,box-shadow 0.25s ease;
  animation:slideUp 0.5s ease both;
}
.stat-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  border-radius:2px 2px 0 0;
}
.stat-card:nth-child(1)::before{background:linear-gradient(90deg,var(--accent),transparent);}
.stat-card:nth-child(2)::before{background:linear-gradient(90deg,var(--accent2),transparent);}
.stat-card:nth-child(3)::before{background:linear-gradient(90deg,var(--green),transparent);}
.stat-card:nth-child(4)::before{background:linear-gradient(90deg,var(--accent3),transparent);}
.stat-card:nth-child(1){animation-delay:.1s}
.stat-card:nth-child(2){animation-delay:.18s}
.stat-card:nth-child(3){animation-delay:.26s}
.stat-card:nth-child(4){animation-delay:.34s}
.stat-card:hover{transform:translateY(-3px);box-shadow:0 12px 40px rgba(0,0,0,0.3);}
.stat-icon{font-size:1.6rem;margin-bottom:8px}
.stat-num{
  font-size:2rem;font-weight:800;line-height:1;
  font-variant-numeric:tabular-nums;
}
.stat-card:nth-child(1) .stat-num{color:var(--accent)}
.stat-card:nth-child(2) .stat-num{color:var(--accent2)}
.stat-card:nth-child(3) .stat-num{color:var(--green)}
.stat-card:nth-child(4) .stat-num{color:var(--accent3)}
.stat-label{font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-top:4px}

@keyframes slideUp{
  from{opacity:0;transform:translateY(20px)}
  to{opacity:1;transform:translateY(0)}
}

/* ── CHATBOX ── */
.chatbox{
  flex:1;
  background:var(--glass);
  border:1px solid var(--border);
  border-radius:20px;
  backdrop-filter:blur(24px);
  display:flex;flex-direction:column;
  overflow:hidden;
  min-height:420px;
  box-shadow:0 8px 48px rgba(0,0,0,0.4);
}
.chat-header{
  padding:16px 20px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:12px;
}

/* Animated bot avatar */
.bot-avatar{
  width:40px;height:40px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;
  font-size:20px;
  position:relative;
  flex-shrink:0;
  box-shadow:0 0 20px rgba(0,170,255,0.4);
  animation:avatarGlow 2.5s ease-in-out infinite alternate;
}
@keyframes avatarGlow{
  from{box-shadow:0 0 20px rgba(0,170,255,0.4);}
  to{box-shadow:0 0 35px rgba(168,85,247,0.5),0 0 60px rgba(0,170,255,0.2);}
}
.bot-avatar::after{
  content:'';
  position:absolute;inset:-3px;
  border-radius:50%;
  border:2px solid transparent;
  border-top-color:var(--accent);
  border-right-color:var(--accent2);
  animation:spin 3s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.bot-status{
  display:flex;flex-direction:column;
}
.bot-name{font-weight:700;font-size:0.95rem}
.bot-online{
  font-size:0.72rem;color:var(--green);
  display:flex;align-items:center;gap:5px;
}
.bot-online::before{
  content:'';width:6px;height:6px;border-radius:50%;
  background:var(--green);
  animation:pulse 1.5s ease-in-out infinite;
}
@keyframes pulse{
  0%,100%{transform:scale(1);opacity:1}
  50%{transform:scale(1.5);opacity:0.5}
}

/* Messages area */
.messages{
  flex:1;overflow-y:auto;padding:20px;
  display:flex;flex-direction:column;gap:16px;
}
.messages::-webkit-scrollbar{width:4px}
.messages::-webkit-scrollbar-thumb{background:rgba(0,170,255,0.2);border-radius:2px}

/* Message bubbles */
.msg{
  display:flex;gap:10px;
  animation:msgIn 0.35s cubic-bezier(0.34,1.56,0.64,1) both;
}
@keyframes msgIn{
  from{opacity:0;transform:translateY(10px) scale(0.96)}
  to{opacity:1;transform:translateY(0) scale(1)}
}
.msg.user{flex-direction:row-reverse}

.msg-avatar{
  width:32px;height:32px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  font-size:14px;flex-shrink:0;
  border:1px solid var(--border);
  background:var(--bg2);
}
.msg.user .msg-avatar{
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  font-size:13px;font-weight:700;color:#fff;
}

.msg-bubble{
  max-width:75%;padding:12px 16px;border-radius:18px;
  font-size:0.875rem;line-height:1.6;
}
.msg.bot .msg-bubble{
  background:var(--bg2);
  border:1px solid var(--border);
  border-radius:4px 18px 18px 18px;
}
.msg.user .msg-bubble{
  background:linear-gradient(135deg,rgba(0,170,255,0.2),rgba(168,85,247,0.15));
  border:1px solid rgba(0,170,255,0.2);
  border-radius:18px 4px 18px 18px;
  color:var(--text);
}
.msg-bubble strong{color:var(--accent)}
.msg-bubble code{
  background:rgba(0,170,255,0.1);color:var(--accent);
  padding:1px 5px;border-radius:4px;
  font-family:'JetBrains Mono',monospace;font-size:0.82rem;
}

/* Typing indicator */
.typing{
  display:flex;gap:10px;
  animation:msgIn 0.3s ease both;
}
.typing-dots{
  display:flex;align-items:center;gap:5px;
  background:var(--bg2);border:1px solid var(--border);
  padding:12px 16px;border-radius:4px 18px 18px 18px;
}
.typing-dots span{
  width:7px;height:7px;border-radius:50%;
  animation:bounce 1.3s ease-in-out infinite;
}
.typing-dots span:nth-child(1){background:var(--accent);animation-delay:0s}
.typing-dots span:nth-child(2){background:var(--accent2);animation-delay:0.15s}
.typing-dots span:nth-child(3){background:var(--accent3);animation-delay:0.3s}
@keyframes bounce{
  0%,80%,100%{transform:translateY(0);opacity:0.5}
  40%{transform:translateY(-6px);opacity:1}
}

/* Tool status */
.tool-status{
  display:flex;align-items:center;gap:8px;
  padding:8px 14px;border-radius:20px;
  font-size:0.78rem;font-weight:500;
  width:fit-content;
  animation:fadeIn 0.3s ease;
}
.tool-status.searching{
  background:rgba(0,170,255,0.1);border:1px solid rgba(0,170,255,0.25);color:var(--accent);
}
.tool-status.wikipedia{
  background:rgba(168,85,247,0.1);border:1px solid rgba(168,85,247,0.25);color:var(--accent2);
}
.tool-status.reading{
  background:rgba(249,115,22,0.1);border:1px solid rgba(249,115,22,0.25);color:var(--accent3);
}
.tool-status::before{
  content:'';width:6px;height:6px;border-radius:50%;background:currentColor;
  animation:pulse 1s ease-in-out infinite;
}
@keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}

/* Input area */
.input-area{
  padding:16px;border-top:1px solid var(--border);
  display:flex;flex-direction:column;gap:10px;
}

/* Quick replies */
.quick-replies{
  display:flex;gap:8px;flex-wrap:wrap;
}
.qr-btn{
  padding:6px 14px;border-radius:20px;font-size:0.76rem;font-weight:500;
  background:transparent;border:1px solid var(--border);color:var(--muted);
  cursor:pointer;transition:all 0.2s ease;font-family:'Outfit',sans-serif;
}
.qr-btn:hover{
  background:rgba(0,170,255,0.08);border-color:rgba(0,170,255,0.3);color:var(--accent);
  transform:translateY(-1px);
}

.input-row{
  display:flex;gap:10px;align-items:flex-end;
}
.msg-input{
  flex:1;background:var(--bg2);border:1px solid var(--border);
  border-radius:14px;padding:12px 16px;
  color:var(--text);font-family:'Outfit',sans-serif;font-size:0.875rem;
  resize:none;min-height:48px;max-height:120px;outline:none;
  transition:border-color 0.25s ease,box-shadow 0.25s ease;
  line-height:1.5;
}
.msg-input:focus{
  border-color:rgba(0,170,255,0.4);
  box-shadow:0 0 0 3px rgba(0,170,255,0.08);
}
.msg-input::placeholder{color:var(--muted)}
.send-btn{
  width:48px;height:48px;border-radius:12px;flex-shrink:0;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  border:none;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  transition:all 0.2s ease;
  box-shadow:0 4px 20px rgba(0,170,255,0.3);
  font-size:18px;
}
.send-btn:hover{transform:scale(1.05);box-shadow:0 6px 28px rgba(0,170,255,0.4);}
.send-btn:active{transform:scale(0.97)}

/* ── RIGHT SIDEBAR ── */
.sidebar{
  display:flex;flex-direction:column;gap:16px;
  animation:slideUp 0.5s ease 0.2s both;
}

/* Laptop illustration card */
.illustration-card{
  background:var(--glass);
  border:1px solid var(--border);
  border-radius:20px;
  padding:20px;
  backdrop-filter:blur(20px);
  overflow:hidden;position:relative;
}
.illustration-card::after{
  content:'';position:absolute;
  bottom:-40px;right:-40px;
  width:120px;height:120px;
  background:radial-gradient(circle,rgba(0,170,255,0.1),transparent 70%);
  border-radius:50%;
}

/* SVG laptop */
.laptop-svg{
  display:block;margin:0 auto 16px;
  filter:drop-shadow(0 8px 24px rgba(0,170,255,0.25));
  animation:laptopFloat 4s ease-in-out infinite;
}
@keyframes laptopFloat{
  0%,100%{transform:translateY(0)}
  50%{transform:translateY(-8px)}
}

.il-title{
  font-weight:700;font-size:0.9rem;text-align:center;margin-bottom:4px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.il-sub{font-size:0.72rem;color:var(--muted);text-align:center}

/* Info cards */
.info-card{
  background:var(--glass);border:1px solid var(--border);
  border-radius:var(--r);padding:16px;backdrop-filter:blur(20px);
}
.info-card h3{
  font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;
  color:var(--muted);margin-bottom:12px;font-weight:600;
}

/* Tier display */
.tiers{display:flex;flex-direction:column;gap:8px}
.tier-item{
  display:flex;align-items:center;gap:10px;
  padding:8px 10px;border-radius:10px;
  background:rgba(255,255,255,0.02);
  border:1px solid transparent;
  transition:all 0.2s ease;cursor:default;
}
.tier-item:hover{background:rgba(255,255,255,0.05);border-color:var(--border)}
.tier-num{
  width:24px;height:24px;border-radius:7px;
  display:flex;align-items:center;justify-content:center;
  font-size:0.75rem;font-weight:700;flex-shrink:0;
  font-family:'JetBrains Mono',monospace;
}
.tier-info{flex:1}
.tier-name{font-size:0.8rem;font-weight:600}
.tier-chips{font-size:0.7rem;color:var(--muted);margin-top:1px}

/* GPU mini illustration */
.gpu-svg{
  display:block;margin:10px auto 0;
  filter:drop-shadow(0 4px 16px rgba(168,85,247,0.3));
  animation:gpuGlow 3s ease-in-out infinite alternate;
}
@keyframes gpuGlow{
  from{filter:drop-shadow(0 4px 16px rgba(168,85,247,0.25))}
  to{filter:drop-shadow(0 4px 24px rgba(168,85,247,0.5))}
}

/* Capabilities */
.cap-list{display:flex;flex-direction:column;gap:8px}
.cap-item{
  display:flex;align-items:center;gap:10px;
  font-size:0.8rem;color:#b0bec5;padding:4px 0;
}
.cap-icon{font-size:1rem;flex-shrink:0}
.cap-dot{
  width:6px;height:6px;border-radius:50%;flex-shrink:0;
  margin-left:auto;
}
.cap-dot.active{background:var(--green);animation:pulse 2s infinite}

/* Export btn */
.export-btn{
  width:100%;padding:10px;border-radius:10px;
  background:transparent;border:1px solid var(--border);
  color:var(--muted);font-family:'Outfit',sans-serif;
  font-size:0.8rem;cursor:pointer;transition:all 0.2s ease;
  display:flex;align-items:center;justify-content:center;gap:6px;
}
.export-btn:hover{
  background:rgba(255,255,255,0.05);border-color:rgba(255,255,255,0.15);color:var(--text);
}

/* Responsive */
@media(max-width:1000px){
  .app{grid-template-columns:1fr;grid-template-rows:auto auto 1fr auto}
  .sidebar{grid-row:4}
  .stats-row{grid-template-columns:repeat(2,1fr)}
}
</style>
</head>
<body>

<canvas id="bg-canvas"></canvas>
<div class="grid-overlay"></div>

<div class="app">

  <!-- HEADER -->
  <header class="header">
    <div class="logo">
      <div class="logo-icon">🖥️</div>
      <div class="logo-text">
        <h1>Laptop Advisor AI</h1>
        <p>Powered by Ollama · Local & Private · 1020 Laptops</p>
      </div>
    </div>
    <div class="header-badges">
      <span class="badge badge-cyan">⚡ Streaming</span>
      <span class="badge badge-purple">🧠 Memory</span>
      <span class="badge badge-green">🔍 Web Search</span>
      <span class="badge badge-orange">📖 Wikipedia</span>
    </div>
  </header>

  <!-- MAIN CHAT COLUMN -->
  <main class="chat-panel">

    <!-- Stat cards -->
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-icon">💾</div>
        <div class="stat-num" id="db-count">1020</div>
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
        <div class="stat-num" id="msg-count">0</div>
        <div class="stat-label">Messages</div>
      </div>
    </div>

    <!-- Chat box -->
    <div class="chatbox">
      <div class="chat-header">
        <div class="bot-avatar">🤖</div>
        <div class="bot-status">
          <span class="bot-name">Laptop Advisor</span>
          <span class="bot-online">Online · Ready to help</span>
        </div>
      </div>

      <div class="messages" id="messages-container">
        <!-- Messages injected by JS -->
      </div>

      <div class="input-area">
        <div class="quick-replies" id="quick-replies">
          <button class="qr-btn" onclick="setInput('Best gaming laptop')">🎮 Best gaming laptop</button>
          <button class="qr-btn" onclick="setInput('Lightweight for college')">🎒 College laptop</button>
          <button class="qr-btn" onclick="setInput('What is GPU tier?')">❓ What is GPU tier?</button>
          <button class="qr-btn" onclick="setInput('Latest RTX laptops 2024')">🔍 Latest RTX</button>
          <button class="qr-btn" onclick="setInput('Compare option 1 and 2')">⚖️ Compare</button>
          <button class="qr-btn" onclick="setInput('Best for ML work')">🧠 ML workstation</button>
        </div>
        <div class="input-row">
          <textarea id="msg-input" class="msg-input"
            placeholder="Ask anything — laptop search, tech questions, paste a URL..."
            rows="1"></textarea>
          <button class="send-btn" id="send-btn" onclick="sendMessage()">➤</button>
        </div>
      </div>
    </div>

  </main>

  <!-- SIDEBAR -->
  <aside class="sidebar">

    <!-- Laptop illustration -->
    <div class="illustration-card">
      <!-- Animated SVG Laptop -->
      <svg class="laptop-svg" width="200" height="130" viewBox="0 0 200 130" fill="none">
        <!-- Screen -->
        <rect x="30" y="10" width="140" height="88" rx="6" fill="#0d1525" stroke="rgba(0,170,255,0.4)" stroke-width="1.5"/>
        <!-- Screen bezel -->
        <rect x="38" y="18" width="124" height="72" rx="3" fill="#080d18"/>
        <!-- Screen glow content -->
        <rect x="42" y="22" width="116" height="8" rx="2" fill="rgba(0,170,255,0.12)"/>
        <rect x="42" y="34" width="80" height="5" rx="2" fill="rgba(168,85,247,0.1)"/>
        <rect x="42" y="43" width="100" height="5" rx="2" fill="rgba(0,170,255,0.08)"/>
        <rect x="42" y="52" width="60" height="5" rx="2" fill="rgba(34,211,160,0.1)"/>
        <rect x="42" y="61" width="90" height="5" rx="2" fill="rgba(0,170,255,0.07)"/>
        <rect x="42" y="70" width="70" height="5" rx="2" fill="rgba(168,85,247,0.08)"/>
        <!-- Cursor blink -->
        <rect x="42" y="82" width="3" height="7" rx="1" fill="var(--accent)">
          <animate attributeName="opacity" values="1;0;1" dur="1.2s" repeatCount="indefinite"/>
        </rect>
        <!-- Base -->
        <rect x="20" y="98" width="160" height="8" rx="4" fill="#0d1525" stroke="rgba(0,170,255,0.3)" stroke-width="1"/>
        <!-- Trackpad -->
        <rect x="80" y="100" width="40" height="5" rx="2.5" fill="rgba(0,170,255,0.1)" stroke="rgba(0,170,255,0.2)" stroke-width="1"/>
        <!-- Bottom edge -->
        <path d="M10 106 Q100 112 190 106 L195 112 Q100 118 5 112 Z" fill="#080d18" stroke="rgba(0,170,255,0.2)" stroke-width="1"/>
        <!-- Glow under screen -->
        <ellipse cx="100" cy="98" rx="55" ry="3" fill="rgba(0,170,255,0.12)"/>
      </svg>
      <div class="il-title">AI-Powered Laptop Advisor</div>
      <div class="il-sub">1020 laptops · CPU & GPU tiers · Local AI</div>
    </div>

    <!-- CPU Tiers -->
    <div class="info-card">
      <h3>CPU Performance Tiers</h3>
      <div class="tiers">
        <div class="tier-item">
          <div class="tier-num" style="background:rgba(0,170,255,0.1);color:var(--accent);border:1px solid rgba(0,170,255,0.2)">1</div>
          <div class="tier-info">
            <div class="tier-name" style="color:var(--accent)">Basic</div>
            <div class="tier-chips">Intel i3 · AMD Ryzen 3 · Celeron</div>
          </div>
        </div>
        <div class="tier-item">
          <div class="tier-num" style="background:rgba(0,170,255,0.2);color:var(--accent);border:1px solid rgba(0,170,255,0.35)">2</div>
          <div class="tier-info">
            <div class="tier-name" style="color:var(--accent)">Mid-range</div>
            <div class="tier-chips">Intel i5 · AMD Ryzen 5</div>
          </div>
        </div>
        <div class="tier-item">
          <div class="tier-num" style="background:rgba(0,170,255,0.32);color:var(--accent);border:1px solid rgba(0,170,255,0.5)">3</div>
          <div class="tier-info">
            <div class="tier-name" style="color:var(--accent)">High-end</div>
            <div class="tier-chips">Intel i7 · AMD Ryzen 7</div>
          </div>
        </div>
        <div class="tier-item">
          <div class="tier-num" style="background:rgba(0,170,255,0.5);color:#fff;border:1px solid rgba(0,170,255,0.7)">4</div>
          <div class="tier-info">
            <div class="tier-name" style="color:var(--accent)">Top-tier</div>
            <div class="tier-chips">Intel i9 · AMD Ryzen 9</div>
          </div>
        </div>
      </div>
    </div>

    <!-- GPU card with SVG -->
    <div class="info-card">
      <h3>GPU Performance Tiers</h3>
      <!-- GPU SVG illustration -->
      <svg class="gpu-svg" width="180" height="60" viewBox="0 0 180 60" fill="none">
        <!-- PCB -->
        <rect x="10" y="15" width="160" height="35" rx="4" fill="#0d1525" stroke="rgba(168,85,247,0.4)" stroke-width="1"/>
        <!-- Heatsink fins -->
        <rect x="20" y="10" width="8" height="20" rx="1" fill="rgba(168,85,247,0.2)" stroke="rgba(168,85,247,0.3)" stroke-width="0.5"/>
        <rect x="32" y="8" width="8" height="22" rx="1" fill="rgba(168,85,247,0.25)" stroke="rgba(168,85,247,0.35)" stroke-width="0.5"/>
        <rect x="44" y="10" width="8" height="20" rx="1" fill="rgba(168,85,247,0.2)" stroke="rgba(168,85,247,0.3)" stroke-width="0.5"/>
        <rect x="56" y="6" width="8" height="24" rx="1" fill="rgba(168,85,247,0.3)" stroke="rgba(168,85,247,0.4)" stroke-width="0.5"/>
        <rect x="68" y="10" width="8" height="20" rx="1" fill="rgba(168,85,247,0.2)" stroke="rgba(168,85,247,0.3)" stroke-width="0.5"/>
        <!-- Chip -->
        <rect x="90" y="22" width="30" height="20" rx="3" fill="rgba(168,85,247,0.2)" stroke="rgba(168,85,247,0.5)" stroke-width="1"/>
        <text x="105" y="35" font-family="monospace" font-size="7" fill="rgba(168,85,247,0.8)" text-anchor="middle">GPU</text>
        <!-- VRAM chips -->
        <rect x="128" y="22" width="10" height="8" rx="1" fill="rgba(168,85,247,0.15)" stroke="rgba(168,85,247,0.3)" stroke-width="0.5"/>
        <rect x="142" y="22" width="10" height="8" rx="1" fill="rgba(168,85,247,0.15)" stroke="rgba(168,85,247,0.3)" stroke-width="0.5"/>
        <rect x="128" y="34" width="10" height="8" rx="1" fill="rgba(168,85,247,0.15)" stroke="rgba(168,85,247,0.3)" stroke-width="0.5"/>
        <rect x="142" y="34" width="10" height="8" rx="1" fill="rgba(168,85,247,0.15)" stroke="rgba(168,85,247,0.3)" stroke-width="0.5"/>
        <!-- PCIe connectors -->
        <rect x="12" y="46" width="140" height="6" rx="1" fill="rgba(168,85,247,0.1)" stroke="rgba(168,85,247,0.2)" stroke-width="0.5"/>
        <!-- Glow -->
        <ellipse cx="90" cy="32" rx="18" ry="8" fill="rgba(168,85,247,0.05)">
          <animate attributeName="opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite"/>
        </ellipse>
      </svg>
      <div class="tiers" style="margin-top:10px">
        <div class="tier-item">
          <div class="tier-num" style="background:rgba(168,85,247,0.08);color:var(--accent2);border:1px solid rgba(168,85,247,0.2)">0</div>
          <div class="tier-info"><div class="tier-name" style="color:var(--accent2)">Integrated</div><div class="tier-chips">Intel UHD · AMD Radeon</div></div>
        </div>
        <div class="tier-item">
          <div class="tier-num" style="background:rgba(168,85,247,0.16);color:var(--accent2);border:1px solid rgba(168,85,247,0.3)">1</div>
          <div class="tier-info"><div class="tier-name" style="color:var(--accent2)">Entry Gaming</div><div class="tier-chips">GTX 1650 · RTX 3050</div></div>
        </div>
        <div class="tier-item">
          <div class="tier-num" style="background:rgba(168,85,247,0.28);color:var(--accent2);border:1px solid rgba(168,85,247,0.45)">2</div>
          <div class="tier-info"><div class="tier-name" style="color:var(--accent2)">Mid Gaming</div><div class="tier-chips">RTX 3060 · RTX 4060</div></div>
        </div>
        <div class="tier-item">
          <div class="tier-num" style="background:rgba(168,85,247,0.45);color:#fff;border:1px solid rgba(168,85,247,0.65)">3</div>
          <div class="tier-info"><div class="tier-name" style="color:var(--accent2)">High-end</div><div class="tier-chips">RTX 3080 · RTX 4080+</div></div>
        </div>
      </div>
    </div>

    <!-- Capabilities -->
    <div class="info-card">
      <h3>What I Can Do</h3>
      <div class="cap-list">
        <div class="cap-item"><span class="cap-icon">🖥️</span>Search 1020 laptop database<div class="cap-dot active"></div></div>
        <div class="cap-item"><span class="cap-icon">🔍</span>DuckDuckGo web search<div class="cap-dot active"></div></div>
        <div class="cap-item"><span class="cap-icon">📖</span>Wikipedia tech lookups<div class="cap-dot active"></div></div>
        <div class="cap-item"><span class="cap-icon">🔗</span>Read any URL you paste<div class="cap-dot active"></div></div>
        <div class="cap-item"><span class="cap-icon">🧠</span>20-turn conversation memory<div class="cap-dot active"></div></div>
        <div class="cap-item"><span class="cap-icon">⚡</span>Real-time streaming<div class="cap-dot active"></div></div>
      </div>
    </div>

    <button class="export-btn" onclick="exportChat()">💾 Export Conversation</button>

  </aside>

</div>

<script>
// ── PARTICLE BACKGROUND ──
const canvas = document.getElementById('bg-canvas');
const ctx2 = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
window.addEventListener('resize', () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; });

const particles = Array.from({length: 60}, () => ({
  x: Math.random() * canvas.width,
  y: Math.random() * canvas.height,
  r: Math.random() * 1.5 + 0.3,
  dx: (Math.random() - 0.5) * 0.3,
  dy: (Math.random() - 0.5) * 0.3,
  color: Math.random() > 0.5 ? '0,170,255' : '168,85,247',
  opacity: Math.random() * 0.4 + 0.1
}));

function drawParticles() {
  ctx2.clearRect(0, 0, canvas.width, canvas.height);
  // Gradient mesh
  const g1 = ctx2.createRadialGradient(canvas.width*0.15, canvas.height*0.2, 0, canvas.width*0.15, canvas.height*0.2, canvas.width*0.4);
  g1.addColorStop(0, 'rgba(0,170,255,0.05)'); g1.addColorStop(1, 'transparent');
  ctx2.fillStyle = g1; ctx2.fillRect(0,0,canvas.width,canvas.height);
  const g2 = ctx2.createRadialGradient(canvas.width*0.85, canvas.height*0.8, 0, canvas.width*0.85, canvas.height*0.8, canvas.width*0.35);
  g2.addColorStop(0, 'rgba(168,85,247,0.06)'); g2.addColorStop(1, 'transparent');
  ctx2.fillStyle = g2; ctx2.fillRect(0,0,canvas.width,canvas.height);

  // Particles
  particles.forEach(p => {
    p.x += p.dx; p.y += p.dy;
    if (p.x < 0) p.x = canvas.width;
    if (p.x > canvas.width) p.x = 0;
    if (p.y < 0) p.y = canvas.height;
    if (p.y > canvas.height) p.y = 0;
    ctx2.beginPath();
    ctx2.arc(p.x, p.y, p.r, 0, Math.PI*2);
    ctx2.fillStyle = `rgba(${p.color},${p.opacity})`;
    ctx2.fill();
  });
  requestAnimationFrame(drawParticles);
}
drawParticles();

// ── CHAT STATE ──
let chatHistory = [];
let msgCount = 0;

// Initial bot message
window.addEventListener('load', () => {
  addBotMessage("Hey! I'm your laptop advisor and tech assistant, powered by local AI.\n\nI've got **1020 laptops** in my database, plus I can search the web, look things up on Wikipedia, and read any URL you share.\n\nHow would you like to start?\n\n**1** → Guide me through 5 questions\n**2** → I'll just describe what I need\n\nOr ask me anything right away!");
});

function addBotMessage(text, isTyping=false) {
  const container = document.getElementById('messages-container');
  if (isTyping) {
    const div = document.createElement('div');
    div.className = 'msg bot'; div.id = 'typing-msg';
    div.innerHTML = `<div class="msg-avatar">🤖</div><div class="typing-dots"><span></span><span></span><span></span></div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return;
  }
  const div = document.createElement('div');
  div.className = 'msg bot';
  div.innerHTML = `<div class="msg-avatar">🤖</div><div class="msg-bubble">${formatMsg(text)}</div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  chatHistory.push({role:'assistant', content: text});
}

function addUserMessage(text) {
  const container = document.getElementById('messages-container');
  const div = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = `<div class="msg-bubble">${escHtml(text)}</div><div class="msg-avatar">U</div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  msgCount++; document.getElementById('msg-count').textContent = msgCount;
  chatHistory.push({role:'user', content: text});
}

function updateLastBotMsg(text) {
  const container = document.getElementById('messages-container');
  const msgs = container.querySelectorAll('.msg.bot:not(#typing-msg)');
  if (msgs.length > 0) {
    const last = msgs[msgs.length-1].querySelector('.msg-bubble');
    if (last) { last.innerHTML = formatMsg(text); container.scrollTop = container.scrollHeight; }
  }
}

function removeTyping() {
  const t = document.getElementById('typing-msg');
  if (t) t.remove();
}

function showTyping() {
  removeTyping();
  addBotMessage('', true);
}

function formatMsg(text) {
  return escHtml(text)
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/`(.+?)`/g,'<code>$1</code>')
    .replace(/\n/g,'<br>');
}
function escHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── SEND MESSAGE ──
async function sendMessage() {
  const input = document.getElementById('msg-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = ''; input.style.height = 'auto';

  addUserMessage(text);
  showTyping();

  // Stream from Gradio backend via fetch
  try {
    const resp = await fetch('/run/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({data: [text, chatHistory.slice(0,-1)]})
    });
    const data = await resp.json();
    removeTyping();
    if (data.data && data.data[1]) {
      const hist = data.data[1];
      const lastBot = hist.filter(m=>m.role==='assistant').pop();
      if (lastBot) {
        const div = document.createElement('div');
        div.className = 'msg bot';
        div.innerHTML = `<div class="msg-avatar">🤖</div><div class="msg-bubble">${formatMsg(lastBot.content)}</div>`;
        document.getElementById('messages-container').appendChild(div);
        document.getElementById('messages-container').scrollTop = 9999;
        chatHistory = hist;
        msgCount++; document.getElementById('msg-count').textContent = msgCount;
      }
    }
  } catch(e) {
    removeTyping();
    addBotMessage('Something went wrong connecting to the AI. Make sure Ollama is running.');
  }
}

function setInput(text) {
  document.getElementById('msg-input').value = text;
  document.getElementById('msg-input').focus();
}

function exportChat() {
  const lines = chatHistory.map(m =>
    `${m.role === 'user' ? 'You' : 'Advisor'}:\n${m.content}\n\n${'─'.repeat(40)}\n`
  ).join('\n');
  const blob = new Blob([`Laptop Advisor Chat\n${'='.repeat(40)}\n\n` + lines], {type:'text/plain'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = `laptop_chat_${Date.now()}.txt`; a.click();
}

// Auto-resize textarea
document.getElementById('msg-input').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

// Enter to send
document.getElementById('msg-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────
# UI — serve HTML + Gradio backend
# ─────────────────────────────────────────────
def build_ui(collection):
    session = {"ob": Onboard(), "mode": "start"}

    def chat(user_msg, history):
        if not user_msg.strip():
            yield "", history
            return

        ob   = session["ob"]
        mode = session["mode"]
        tool = route_query(user_msg, mode)
        history = history + [tm("user", user_msg)]

        if mode == "start" and tool not in ("web_search","wikipedia","url_fetch","general"):
            c = user_msg.strip()
            if c == "1":
                session["mode"] = "guided"; session["ob"] = Onboard(); ob = session["ob"]
                bot = f"Great! Let's find your perfect laptop.\n\nQuestion 1 of {len(GUIDED_QUESTIONS)}\n\n{GUIDED_QUESTIONS[0]}"
                history = history + [tm("assistant", bot)]; yield "", history; return
            elif c == "2":
                session["mode"] = "free"
                bot = "Of course! Just describe what you're looking for — use case, RAM, screen size, brand, anything. Or ask me a general tech question!"
                history = history + [tm("assistant", bot)]; yield "", history; return
            else:
                session["mode"] = "free"

        if session["mode"] == "guided" and not ob.done:
            ob.answer(user_msg); ack = ob.ack()
            if not ob.done:
                bot = f"{ack}\n\nQuestion {ob.qi + 1} of {len(GUIDED_QUESTIONS)}\n\n{ob.q()}"
                history = history + [tm("assistant", bot)]; yield "", history; return
            else:
                hits = retrieve(collection, ob.query()); ctx = build_ctx(hits)
                msgs = bmsgs(REC_SYS, history[:-1], f"User preferences:\n{ob.summary()}\n\nAvailable laptops:\n{ctx}\n\nGive 3 recommendations.")
                session["mode"] = "done"
                prefix = f"{ack}\n\nSearching {collection.count()} laptops...\n\nYour Top Matches\n\nBased on: {ob.summary()}\n\n"
                history = history + [tm("assistant", prefix)]; ft = ""
                for ft in stream_llm(msgs): history[-1] = tm("assistant", prefix + ft); yield "", history
                history[-1] = tm("assistant", prefix + ft + "\n\nAsk me anything as a follow-up!")
                yield "", history; return

        if tool == "web_search":
            history = history + [tm("assistant", "🔍 Searching the web...")]; yield "", history
            results = web_search(user_msg); ctx = fmt_results(results)
            msgs = bmsgs(SYS, history[:-1], f"Question: {user_msg}\n\nWeb results:\n{ctx}\n\nAnswer naturally.")
            ft = ""
            for ft in stream_llm(msgs): history[-1] = tm("assistant", ft); yield "", history
            return

        if tool == "wikipedia":
            history = history + [tm("assistant", "📖 Looking that up...")]; yield "", history
            term = re.sub(r"(what is|what are|explain|how does|define)","",user_msg,flags=re.IGNORECASE).strip()
            summary = wikipedia_summary(term)
            if not summary: results = web_search(user_msg); summary = fmt_results(results)
            msgs = bmsgs(SYS, history[:-1], f"Question: {user_msg}\n\nReference:\n{summary}\n\nExplain simply.")
            ft = ""
            for ft in stream_llm(msgs): history[-1] = tm("assistant", ft); yield "", history
            return

        if tool == "url_fetch":
            url = re.search(r"https?://\S+", user_msg).group(0)
            history = history + [tm("assistant", f"📄 Reading {url}...")]; yield "", history
            page = fetch_page(url)
            msgs = bmsgs(SYS, history[:-1], f"User: {user_msg}\n\nPage:\n{page}\n\nAnswer based on this.")
            ft = ""
            for ft in stream_llm(msgs): history[-1] = tm("assistant", ft); yield "", history
            return

        hits = retrieve(collection, user_msg); ctx = build_ctx(hits)
        uc = (f"{user_msg}\n\nLaptops:\n{ctx}\n\nCompare clearly." if tool == "comparison"
              else f"{user_msg}\n\nRelevant laptops:\n{ctx}")
        msgs = bmsgs(SYS, history[:-1], uc)
        history = history + [tm("assistant","")]
        for ft in stream_llm(msgs): history[-1] = tm("assistant", ft); yield "", history

    def reset():
        session["ob"] = Onboard(); session["mode"] = "start"
        return [tm("assistant", "Hey! I'm your laptop advisor. Type **1** for guided questions or **2** to search freely!")], ""

    with gr.Blocks(title="Laptop Advisor AI") as demo:
        # Serve the custom HTML as the main page
        gr.HTML(DASHBOARD_HTML)

        # Hidden Gradio components for the backend — the JS calls /run/chat
        with gr.Row(visible=False):
            chatbot  = gr.Chatbot()
            msg_box  = gr.Textbox()
            send_btn = gr.Button()
            rst_btn  = gr.Button()

        send_btn.click(chat, [msg_box, chatbot], [msg_box, chatbot])
        msg_box.submit(chat, [msg_box, chatbot], [msg_box, chatbot])
        rst_btn.click(reset, outputs=[chatbot, msg_box])

    return demo

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv or not os.path.exists("./chroma_db")
    if rebuild:
        print("[Setup] Building vector DB..."); collection = build_vectorstore(load_data(CSV_PATH))
    else:
        print("[Setup] Loading vector DB...")
        try: collection = load_vectorstore()
        except: print("[Setup] Rebuilding..."); collection = build_vectorstore(load_data(CSV_PATH))

    print(f"\n[Ollama] {OLLAMA_MODEL}")
    print("[UI] http://localhost:7860\n")
    demo = build_ui(collection)
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)