"""Stricter retrieval eval against the real Chroma index (no LLM calls).
Ground truth comes from phase1_3 scheme_facts.jsonl (the scraped source of truth), not hand-written.
A hit requires the retrieved chunk to be (a) from the right scheme AND (b) contain the true value.
Run with the repo venv:  python evals/run_retrieval_eval.py
"""
import json, re, pathlib, collections
import sqlite3, numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
ROOT = pathlib.Path(__file__).resolve().parent.parent
facts = [json.loads(l) for l in open(ROOT/"phase1_3/reports/extracted/scheme_facts.jsonl") if l.strip()]
# The stored Chroma index was written by a newer chromadb than requirements.txt pins, so read the
# chunk documents + metadata straight from its sqlite file and do exact cosine search with the same model.
_c = sqlite3.connect(str(ROOT/"phase1_4_3/chroma_store/chroma.sqlite3"))
_d = {}
for i,k,v in _c.execute("select id,key,string_value from embedding_metadata where key in ('chroma:document','scheme_name')"):
    _d.setdefault(i,{})[k]=v
DOCS=[_d[i]["chroma:document"] for i in sorted(_d)]; METAS=[{"scheme_name":_d[i].get("scheme_name","")} for i in sorted(_d)]
# Model was saved by a newer sentence-transformers than pip can install here; its config is
# plain CLS pooling + L2 normalise, so load it with transformers directly.
_tok = AutoTokenizer.from_pretrained(str(ROOT/"models/bge-small-en")); _m = AutoModel.from_pretrained(str(ROOT/"models/bge-small-en")).eval()
def encode(texts, bs=32):
    out=[]
    for i in range(0,len(texts),bs):
        b=_tok(texts[i:i+bs],padding=True,truncation=True,max_length=512,return_tensors="pt")
        with torch.no_grad(): h=_m(**b).last_hidden_state[:,0]
        out.append(torch.nn.functional.normalize(h,dim=-1).numpy())
    return np.vstack(out)
DOC_EMB = encode(DOCS)
norm = lambda s: re.sub(r"\s+"," ",re.sub(r"direct (plan )?growth","",s.lower())).strip()

def val_forms(f, field):
    if field=="expense_ratio": v=f["expense_ratio_percent"]; return [f"{v}%"]
    if field=="minimum_sip": v=f["min_sip_amount_inr"]; return [f"₹{v}", f"₹{v:,}"]
    if field=="nav": v=f["nav_value"]; return [f"{v}"]
    if field=="aum": v=f["aum_value_cr"]; return [f"{v:,.2f}"]
    if field=="benchmark": return [f["benchmark_full_name"].lower()]
    if field=="riskometer": return [f["riskometer_label"].replace("_"," ")]
    return []
TEMPL = {
 "expense_ratio": ["What is the expense ratio of {s}?", "{s} expense ratio kitna hai?", "how much does {s} charge as fees (TER)?"],
 "minimum_sip":   ["What is the minimum SIP amount for {s}?", "{s} me minimum SIP kitna hai?", "smallest monthly investment allowed in {s}"],
 "nav":           ["What is the NAV of {s}?", "{s} ka NAV kya hai?", "current price per unit of {s}"],
 "aum":           ["What is the fund size (AUM) of {s}?", "{s} AUM kitna hai?", "how big is {s} in crores"],
 "benchmark":     ["What is the benchmark of {s}?", "{s} kis index ko benchmark karta hai?", "which index is {s} compared against"],
 "riskometer":    ["What is the riskometer level of {s}?", "{s} kitna risky hai riskometer ke hisaab se?", "risk label for {s}"],
}
K=6
res=collections.defaultdict(list)
detail=[]
for f in facts:
    s=f["scheme_name"]
    for field,tmpls in TEMPL.items():
        forms=val_forms(f,field)
        if not forms or f["field_statuses"].get({"minimum_sip":"minimum_sip"}.get(field,field))!="available": continue
        for ti,t in enumerate(tmpls):
            q=t.format(s=s)
            emb=encode([q]).tolist()
            top=np.argsort(-(DOC_EMB@np.array(emb[0])))[:K]
            r={"documents":[[DOCS[j] for j in top]],"metadatas":[[METAS[j] for j in top]]}
            first_scheme=first_answer=None
            for rank,(d,m) in enumerate(zip(r["documents"][0],r["metadatas"][0]),1):
                right=norm(m.get("scheme_name",""))==norm(s)
                if right and first_scheme is None: first_scheme=rank
                if right and any(x.lower() in d.lower() for x in forms) and first_answer is None: first_answer=rank
            kind=["en","hinglish","paraphrase"][ti]
            res[kind].append((first_scheme,first_answer)); res["all"].append((first_scheme,first_answer)); res[field].append((first_scheme,first_answer))
            if first_answer is None: detail.append((s,field,kind,q,first_scheme))
def rate(lst,k,idx): return sum(1 for x in lst if x[idx] and x[idx]<=k)/len(lst)
print("n queries:",len(res["all"]))
print(f'{"slice":12s} n   scheme@1 scheme@6  answer@1 answer@3 answer@6')
for key in ["all","en","hinglish","paraphrase"]+list(TEMPL):
    l=res[key]; print(f'{key:12s} {len(l):3d} {rate(l,1,0):8.0%} {rate(l,6,0):8.0%} {rate(l,1,1):8.0%} {rate(l,3,1):8.0%} {rate(l,6,1):8.0%}')
print("\nMisses (answer-bearing chunk not in top-6):",len(detail))
for d in detail[:40]: print(" ",d)
