# -*- coding: utf-8 -*-
"""
Entrena un clasificador local (LogisticRegression) para etiquetar tokens como:
- DRUG        (fármaco)
- SENSITIVE   (entidad sensible que sí debe anonimizarse)
- OTHER       (cualquier otro token)

Entrada esperada (CSV):
- Por defecto, lee un CSV con columnas:
  text  -> texto original (línea por muestra)
  drug_lexicon (opcional) → ruta a lexicón para weak-labeling (ver CLI)
También acepta ya un CSV anotado con columnas 'token','label' (modo --pretokenized).

Salida:
- Modelo serializado en: profiles/models/drug_token_clf.joblib
"""

import argparse, re, csv, joblib, os
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

UNITS = {"mg","mcg","µg","ug","g","ui","u.i.","ml","mL","cc"}

def load_lexicon(path: str) -> set:
    lex = set()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip().lower()
                if t:
                    lex.add(t)
    return lex

TOKEN_RE = re.compile(r"\w+([\-\.]\w+)*", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return [m.group(0) for m in TOKEN_RE.finditer(text or "")]

def char_ngram_feats(tok: str, n=(3,4,5)):
    s = tok.lower()
    feats = {}
    for k in n:
        for i in range(len(s)-k+1):
            feats[f"char{ k }={ s[i:i+k] }"] = 1
    return feats

def context_feats(tokens: List[str], i: int):
    f = {}
    prev1 = tokens[i-1].lower() if i-1 >= 0 else ""
    next1 = tokens[i+1].lower() if i+1 < len(tokens) else ""
    prev2 = tokens[i-2].lower() if i-2 >= 0 else ""
    next2 = tokens[i+2].lower() if i+2 < len(tokens) else ""

    f["prev1"] = prev1
    f["next1"] = next1
    f["prev2"] = prev2
    f["next2"] = next2

    # rasgos de dosis cercanas
    f["ctx_has_unit"] = int(next1 in UNITS or prev1 in UNITS or next2 in UNITS or prev2 in UNITS)
    f["ctx_has_number"] = int(bool(re.search(r"\d", next1) or re.search(r"\d", prev1)))
    f["is_capitalized"] = int(tokens[i][:1].isupper())

    return f

def featurize(tokens: List[str], lex: set):
    feats = []
    for i, tok in enumerate(tokens):
        f = {}
        f.update(char_ngram_feats(tok))
        f.update(context_feats(tokens, i))
        f["in_drug_lex"] = int(tok.lower() in lex)
        feats.append(f)
    return feats

def weak_label(tokens: List[str], lex: set):
    labels = []
    for i, tok in enumerate(tokens):
        t = tok.lower()
        if t in lex:
            labels.append("DRUG")
        else:
            labels.append("OTHER")
    return labels

def train_from_texts(df: pd.DataFrame, lex: set):
    X, y = [], []
    for _, row in df.iterrows():
        text = row["text"]
        tokens = tokenize(text)
        feats = featurize(tokens, lex)
        labels = weak_label(tokens, lex)  # weak labels; SENSITIVE se puede añadir si anotas
        X.extend(feats)
        y.extend(labels)
    return X, y

def train_from_pretokenized(df: pd.DataFrame, lex: set):
    # Espera columnas: token,label,prev1,next1 (opcionales prev2,next2)
    X, y = [], []
    for _, row in df.iterrows():
        tok = str(row["token"])
        # construir una "oración" simulada para features de contexto
        tokens = [str(row.get("prev2","")), str(row.get("prev1","")), tok, str(row.get("next1","")), str(row.get("next2",""))]
        feats = featurize(tokens, lex)[2]  # el del centro
        X.append(feats)
        y.append(str(row["label"]).upper())
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Ruta al CSV de entrenamiento")
    ap.add_argument("--drug-lexicon", default="profiles/drug_lexicon.txt", help="Lexicón de fármacos para weak labeling / features")
    ap.add_argument("--pretokenized", action="store_true", help="Si el CSV ya trae columnas token,label,...")
    ap.add_argument("--out", default="profiles/models/drug_token_clf.joblib", help="Salida del modelo")
    args = ap.parse_args()

    Path("profiles/models").mkdir(parents=True, exist_ok=True)
    lex = load_lexicon(args.drug_lexicon)
    df = pd.read_csv(args.csv)

    if args.pretokenized:
        X, y = train_from_pretokenized(df, lex)
    else:
        if "text" not in df.columns:
            raise SystemExit("CSV debe tener columna 'text' o usar --pretokenized con 'token','label'")
        X, y = train_from_texts(df, lex)

    # Pipeline: DictVectorizer + LogisticRegression
    pipe = Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))

    joblib.dump({"pipe": pipe, "lex": list(lex)}, args.out)
    print(f"[OK] Modelo guardado en {args.out}")

if __name__ == "__main__":
    main()
