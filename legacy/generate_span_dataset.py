# -*- coding: utf-8 -*-
"""
Genera dataset débil de spans (DRUG / SENSITIVE / OTHER) a partir de:
- textos (train_texts.csv con columna 'text')
- lexicón de fármacos (profiles/drug_lexicon.txt)
- listas básicas (opcionales) de nombres y hospitales para SENSITIVE
Salida: data/spans_weak.jsonl con campos: left, span, right, label
"""

import re, json, random
from pathlib import Path
import pandas as pd

TOKEN = re.compile(r"\w+([\-\.]\w+)*", re.UNICODE)

def load_list(path):
    s = set()
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t:
                s.add(t.lower())
    return s

def tokenize(s):
    return [m.group(0) for m in TOKEN.finditer(s or "")]

def iter_spans(text, window=50):
    for m in TOKEN.finditer(text):
        s, e = m.span()
        left = text[max(0, s-window):s]
        span = m.group(0)
        right = text[e: e+window]
        yield left, span, right

def main():
    Path("data").mkdir(exist_ok=True, parents=True)
    df = pd.read_csv("train_texts.csv")  # columna 'text'
    drugs = load_list("profiles/drug_lexicon.txt")
    names = load_list("profiles/nombres_es.txt")     # opcional
    hosps = load_list("profiles/hospitales_es.txt")  # opcional

    out = []
    for text in df["text"].astype(str):
        for left, span, right in iter_spans(text):
            lw = span.lower()
            if lw in drugs:
                label = "DRUG"
            elif lw in names or lw in hosps:
                label = "SENSITIVE"
            else:
                # heurística: si está junto a dosis, sube prob de DRUG
                ctx = (left[-20:] + " " + right[:20]).lower()
                if re.search(r"\b\d+(?:[.,]\d+)?\s*(mg|mcg|µg|ug|g|ui|ml|mL|cc)\b", ctx):
                    # si no es obvio sensitive, marca DRUG débil
                    label = "DRUG"
                else:
                    label = "OTHER"
            out.append({"left": left, "span": span, "right": right, "label": label})

    # balanceo simple para no inundar con OTHER
    random.shuffle(out)
    others = [x for x in out if x["label"]=="OTHER"][: min(5000, int(len(out)*0.5))]
    drugs_ = [x for x in out if x["label"]=="DRUG"]
    sens_  = [x for x in out if x["label"]=="SENSITIVE"]
    final = drugs_ + sens_ + others
    random.shuffle(final)

    with open("data/spans_weak.jsonl", "w", encoding="utf-8") as f:
        for row in final:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] data/spans_weak.jsonl · DRUG={len(drugs_)}, SENSITIVE={len(sens_)}, OTHER={len(others)}")

if __name__ == "__main__":
    main()
