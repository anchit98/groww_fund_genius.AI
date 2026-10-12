"""End-to-end *deterministic* eval of the production answer path (no LLM, no vector DB).
In backend/server.py a known scheme+field is answered from structured facts (scheme_facts.jsonl) chosen by
rule-based _extract_scheme/_extract_fields; vector retrieval is only supporting context. This runs those
real methods (extracted from server.py by AST) on generated queries and checks the facts-hint returned.
"""
import ast, json, re, pathlib, collections
ROOT = pathlib.Path(__file__).resolve().parent.parent
src = (ROOT/"backend/server.py").read_text()
cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name=="QueryEngine")
keep = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"_normalize_text","_category_tokens","_extract_scheme","_extract_fields","_fact_answer"}]
mod = ast.Module(body=[ast.ClassDef(name="E", bases=[], keywords=[], body=keep, decorator_list=[])], type_ignores=[])
ns = {"re": re, "Optional": __import__("typing").Optional, "List": list, "Dict": dict, "Any": object}
exec(compile(ast.fix_missing_locations(mod), "server_extract", "exec"), ns)
E = ns["E"]; e = E()
facts = {}
for l in open(ROOT/"phase1_3/reports/extracted/scheme_facts.jsonl"):
    if l.strip(): r=json.loads(l); facts[r["scheme_name"]]=r
e.facts = facts

FIELDS = {  # field -> (templates, fact key used to check the returned value)
 "expense_ratio": (["What is the expense ratio of {s}?","{s} expense ratio kitna hai?","how much does {s} charge as fees (TER)?"], "expense_ratio_percent"),
 "minimum_sip":   (["What is the minimum SIP amount for {s}?","{s} me minimum SIP kitna hai?","smallest monthly investment allowed in {s}"], "min_sip_amount_inr"),
 "nav":           (["What is the NAV of {s}?","{s} ka NAV kya hai?","current price per unit of {s}"], "nav_value"),
 "aum":           (["What is the fund size (AUM) of {s}?","{s} AUM kitna hai?","how big is {s} in crores"], "aum_value_cr"),
 "benchmark":     (["What is the benchmark of {s}?","{s} kis index ko benchmark karta hai?","which index is {s} compared against"], "benchmark_full_name"),
 "riskometer":    (["What is the riskometer level of {s}?","{s} kitna risky hai riskometer ke hisaab se?","risk label for {s}"], "riskometer_label"),
}
KIND = ["english","hinglish","paraphrase"]
def mutations(name):
    base=name.replace(" Direct Growth","")
    yield "canonical", name
    yield "no-plan-suffix", base
    yield "lowercase", base.lower()
    yield "compact (midcap/flexicap)", re.sub(r"(Flexi|Multi|Mid|Small|Large) Cap", lambda m:m.group(1)+"cap", base)
    yield "without AMC prefix", base.replace("Quant ","")

stats = collections.defaultdict(lambda:[0,0,0,0])  # n, scheme_ok, field_ok, value_ok
errs=[]
for sname,row in facts.items():
    for field,(tmpls,key) in FIELDS.items():
        if row["field_statuses"].get(field)!="available": continue
        for ti,t in enumerate(tmpls):
            for mname,sv in mutations(sname):
                q=t.format(s=sv)
                got=e._extract_scheme(q); fl=e._extract_fields(q)
                s_ok = got==sname
                f_ok = field in fl
                hint = e._fact_answer(got,field) if got else None
                v_ok = bool(hint) and str(row[key]).lower().replace("_"," ") in hint.lower().replace("_"," ")
                for k in ("all",f"field:{field}",f"lang:{KIND[ti]}",f"name:{mname}"):
                    st=stats[k]; st[0]+=1; st[1]+=s_ok; st[2]+=f_ok; st[3]+=(s_ok and f_ok and v_ok)
                if not (s_ok and f_ok and v_ok): errs.append((sname,field,KIND[ti],mname,q,got,fl))
print(f'{"slice":34s} n    scheme  field  end-to-end-fact-correct')
for k in ["all"]+sorted(k for k in stats if k!="all"):
    n,s,f,v=stats[k]; print(f"{k:34s} {n:4d} {s/n:6.0%} {f/n:6.0%} {v/n:8.0%}")
print("\nFailures:",len(errs))
byc=collections.Counter((x[1],x[3] if x[5]!=x[0] else "-") for x in errs)
for x in errs[:25]: print(" ",x)
