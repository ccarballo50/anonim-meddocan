# -*- coding: utf-8 -*-
"""
ANONIM (Excel) — versión optimizada y conservadora
- Sin modelo secundario (más rápido y menos sobre-anonimización)
- Personas (PER) y ubicaciones administrativas (GPE/LOC) en texto libre
- NO anonimiza hospitales ni organizaciones (ORG/FAC) de forma automática
- Columnas estructuradas: keep/num/cat (conservar), date->año, pii->[X]
- Procesamiento por lotes (spaCy pipe) para mejorar rendimiento
- Auditoría y preview

Requisitos mínimos (entorno de build/venv):
  pip install spacy==3.8.7 pandas pyyaml openpyxl
"""

from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import argparse
import json
import re
import shutil
import pandas as pd

try:
    import yaml
except Exception:
    raise SystemExit("Falta PyYAML. Instala con: pip install pyyaml")

try:
    import spacy
except Exception:
    raise SystemExit("Falta spaCy. Instala con: pip install spacy==3.8.7")


# ============================
# CONFIG DE RENDIMIENTO Y POLÍTICA
# ============================

# Modelo secundario (desactivado por petición del usuario)
USE_SECONDARY_NER = False          # << NO usar 'es_core_news_md' por ahora
SECONDARY_ONLY_LOC_GPE = True      # (sin efecto si USE_SECONDARY_NER=False)
SECONDARY_ONLY_WHEN_NO_GEO = True  # (sin efecto si USE_SECONDARY_NER=False)

# Lotes y procesos
BATCH_SIZE = 512   # Ajusta según RAM/CPU (256–1024)
N_PROC = 0         # 0 = auto (1 core). Probar 2–4 si se dispone.

# Etiquetas NER permitidas (conservador)
# - Personas SIEMPRE
# - Ubicaciones administrativas: GPE/LOC
# - NO tapar ORG/FAC (evita hospitales por defecto) ni NORP/MISC
ALLOWED_MAIN_LABELS = {"PER", "GPE", "LOC"}

# Heurística/regex: lista blanca de términos sanitarios NO PII
SAFE_ORG_TERMS = {
    "urgencias", "servicio de urgencias",
    "médico de atención primaria", "medico de atencion primaria",
    "atención primaria", "atencion primaria",
    "samur", "summa", "uvi", "uvi móvil", "uvis", "suap", "pac"
}

# Sufijos típicos farmacológicos y ejemplos frecuentes (para NO tapar)
DRUG_SUFFIXES = (
    "ina","pril","sartan","mab","cillin","micina","zida","zolam","zepam","prazol",
    "cort","metazona","metasona","olol","dipina","caina","statin","statina","zolina",
    "zepina","setron","tromb","acetil","morfina","paracetamol","ibuprofeno","omeprazol",
    "metformina","atorvastatina","enoxaparina","clopidogrel","aspirina","adiro","heparina"
)

def is_drug_name(text: str) -> bool:
    t = (text or "").strip().lower()
    if len(t) <= 2:
        return False
    toks = t.split()
    for w in toks:
        if any(w.endswith(s) for s in DRUG_SUFFIXES):
            return True
    return any(t.endswith(s) for s in DRUG_SUFFIXES)

def is_generic_acronym(text: str) -> bool:
    t = (text or "").strip()
    return t.isupper() and 3 <= len(t) <= 8

# ============================
# CARGA DE MODELO PRINCIPAL (MedDOCAN u otro entrenado)
# ============================
_NLP_MAIN = None

def load_main_nlp(model_path: str):
    global _NLP_MAIN
    if _NLP_MAIN is None:
        _NLP_MAIN = spacy.load(str(Path(model_path)))
    return _NLP_MAIN

# ============================
# PERFIL: normalización y utilidades
# ============================
def _get_first(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] not in (None, [], {}):
            return d[k]
    return default

def _load_gazetteer_from_files(paths):
    terms = set()
    for p in (paths or []):
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    t = line.strip()
                    if t:
                        terms.add(t)
        except Exception:
            pass
    return terms

def normalize_profile(profile: dict):
    """
    Devuelve:
      - text_cols: lista de columnas de texto libre
      - stypes: dict {col: kind}
      - ner_labels: etiquetas a considerar (se ignoran en ORG/FAC en código)
      - regex_map: patrones complementarios
      - gazetteer: set de términos a reemplazar como [LOC]
      - masking_strategy: 'tag' (default) o 'redact'
      - tag_map: mapa para etiquetas NER -> tokens
      - model_path: ruta modelo spaCy principal
    """
    text_cols = _get_first(profile, "text_columns", "text_cols", default=[]) or []
    stypes = _get_first(profile, "structured_types", default={}) or {}
    if not stypes:
        scols = _get_first(profile, "structured_cols", default=[]) or []
        if isinstance(scols, list):
            stypes = {
                it["column"]: it["kind"]
                for it in scols
                if isinstance(it, dict) and "column" in it and "kind" in it
            }

    # Lista por defecto: PER/GPE/LOC (conservador). ORG/FAC NO (para evitar hospitales).
    ner_labels = _get_first(profile, "ner_labels", default=["PER", "GPE", "LOC"])

    regex_map = _get_first(profile, "regex", default={}) or {}

    gaz_cities = set(_get_first(profile, "gazetteer_cities", default=[]) or [])
    gaz_files = _get_first(profile, "gazetteer_files", default=[]) or []
    gaz_from_files = _load_gazetteer_from_files(gaz_files)
    gazetteer = gaz_cities.union(gaz_from_files)

    masking = _get_first(profile, "masking", default={}) or {}
    masking_strategy = masking.get("strategy", "tag")
    tag_map = masking.get("tag_map", {}) or {
        "PER": "[NOMBRE]",
        "GPE": "[LOC]",
        "LOC": "[LOC]"
    }

    model_path = profile.get("model_path", "models/ner_meddocan/model-best")

    return {
        "text_cols": text_cols,
        "stypes": stypes,
        "ner_labels": ner_labels,
        "regex_map": regex_map,
        "gazetteer": gazetteer,
        "masking_strategy": masking_strategy,
        "tag_map": tag_map,
        "model_path": model_path
    }

# ============================
# ESTRUCTURADAS: helpers
# ============================
def mask_full(value, strategy="tag", token="[X]"):
    if pd.isna(value):
        return value
    if strategy == "tag":
        return token
    s = str(value)
    return "█" * (len(s) if s else 1)

def anonymize_date(value):
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return value
    ts = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(ts):
        return ""  # o "[FECHA]"
    return int(ts.year)

def anonymize_struct_value(val, kind, regex_cfg: dict, masking_strategy="tag"):
    """
    - 'pii'  -> [X]
    - 'date' -> año
    - 'num','cat','keep' -> conservar
    - otros (dni/nie/nhc/phone/email/address/name) -> regex; si no casa, [X]
    """
    if pd.isna(val):
        return val, 0

    if kind in ("num", "cat", "keep"):
        return val, 0

    if kind == "date":
        return anonymize_date(val), 1

    if kind == "pii":
        return mask_full(val, strategy=masking_strategy, token="[X]"), 1

    tag_by_kind = {
        "dni": "[DNI]", "nie": "[NIE]", "nhc": "[ID]",
        "phone": "[TEL]", "email": "[EMAIL]",
        "address": "[CALLE]", "name": "[NOMBRE]"
    }
    tag = tag_by_kind.get(kind, "[X]")
    pattern = regex_cfg.get(kind)
    s = str(val)
    if not pattern:
        return mask_full(s, strategy=masking_strategy, token="[X]"), 1

    new, n = re.subn(pattern, tag, s, flags=re.IGNORECASE)
    if n == 0:
        return mask_full(s, strategy=masking_strategy, token="[X]"), 1
    return new, n
# ============================
# ANONIMIZACIÓN DE TEXTO EN LOTE (PIPE CON MODELO PRINCIPAL)
# ============================

def _compile_regex_bundle(regex_cfg: dict, gazetteer: set):
    """Compila patrones regex una sola vez para velocidad."""
    def comp(pat):
        return re.compile(pat, re.IGNORECASE) if pat else None

    rx = {
        "dni": comp(regex_cfg.get("dni")),
        "nie": comp(regex_cfg.get("nie")),
        "phone": comp(regex_cfg.get("phone")),
        "email": comp(regex_cfg.get("email")),
        "date": comp(regex_cfg.get("date")),
        "nhc": comp(regex_cfg.get("nhc")),
        "vive": re.compile(r"\b(vive en|reside en|domiciliad[oa] en)\s+[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑ\-]+", re.IGNORECASE)
    }
    gaz_pat = None
    if gazetteer:
        gaz_pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in gazetteer if t) + r")\b", re.IGNORECASE)
    return rx, gaz_pat

def _apply_regex_all(text: str, rx, gaz_pat):
    """Aplica DNI/NIE/PHONE/... + gazetteer + heurística en texto final."""
    if not text:
        return text
    out = text
    def sub(p, token):
        nonlocal out
        if p:
            out = p.sub(token, out)
    sub(rx["dni"],   "[DNI]")
    sub(rx["nie"],   "[NIE]")
    sub(rx["phone"], "[TEL]")
    sub(rx["email"], "[EMAIL]")
    sub(rx["date"],  "[FECHA]")
    if rx["nhc"]:
        sub(rx["nhc"], "[ID]")
    if gaz_pat:
        out = gaz_pat.sub("[LOC]", out)
    # Heurística: "vive en <topónimo>" -> [LOC]
    out = rx["vive"].sub(r"\1 [LOC]", out)
    return out

def _apply_spans(text: str, spans, tag_map, masking_strategy="tag"):
    """Reemplaza spans [ (start,end,label), ... ] en orden inverso."""
    if not spans:
        return text, {}
    # ordenar y aplicar de derecha a izquierda
    spans = sorted(spans, key=lambda x: x[0], reverse=True)
    out = text
    counts = {}
    for s, e, lab in spans:
        repl = tag_map.get(lab, f"[{lab}]") if masking_strategy == "tag" else "█"
        out = out[:s] + repl + out[e:]
        counts[lab] = counts.get(lab, 0) + 1
    return out, counts

def anonymize_texts(
    texts: List[str],
    nlp_main,
    tag_map,
    regex_cfg,
    ner_labels,
    gazetteer,
    masking_strategy="tag"
):
    """
    - Lotes con nlp_main.pipe
    - Filtra entidades conservadoramente (PER, GPE, LOC)
    - Aplica regex/gazetteer/heurística
    """
    rx, gaz_pat = _compile_regex_bundle(regex_cfg, gazetteer)
    final = []
    summary = {"ner": {}, "regex": {}, "gazetteer": 0, "heur": 0}

    with nlp_main.select_pipes(enable=["ner"]):
        for doc in nlp_main.pipe(texts, batch_size=BATCH_SIZE, n_process=N_PROC):
            text_orig = doc.text
            # spans para PER/GPE/LOC
            spans = []
            for ent in doc.ents:
                lab = ent.label_
                if lab not in ner_labels:  
                    continue
                if lab == "PER":
                    spans.append((ent.start_char, ent.end_char, lab))
                elif lab in {"GPE", "LOC"}:
                    spans.append((ent.start_char, ent.end_char, lab))
            replaced, cts = _apply_spans(text_orig, spans, tag_map, masking_strategy)
            for k, v in cts.items():
                summary["ner"][k] = summary["ner"].get(k, 0) + v

            out2 = _apply_regex_all(replaced, rx, gaz_pat)
            final.append(out2)

    return final, summary

# ============================
# WIZARD: creación de perfil interactivo
# ============================
HEUR_TEXT = re.compile(r"(texto|nota|anamnesis|evoluci[oó]n|informe|descrip|historia)", re.I)
HEUR_STRUCT = re.compile(r"(dni|nie|nhc|historia|id|tel[eé]fono|tel|m[oó]vil|email|correo|direcci[oó]n|calle|pais|cp|c[oó]digo_postal|fecha|hora)", re.I)

def suggest_mapping(df: pd.DataFrame):
    text_cols, struct_cols = [], []
    for c in df.columns:
        sample = df[c].dropna().astype(str).head(50)
        avg_len = (sample.map(len).mean() if not sample.empty else 0)
        if HEUR_TEXT.search(c) or avg_len > 150:
            text_cols.append(c)
        elif HEUR_STRUCT.search(c) or avg_len < 40:
            struct_cols.append(c)
    return text_cols, struct_cols

def cli_wizard_build_profile(xlsx_path: str, profile_name: str, default_model="models/ner_meddocan/model-best"):
    df = pd.read_excel(xlsx_path)
    print(f"\nColumnas detectadas: {list(df.columns)}")

    tc, sc = suggest_mapping(df)
    print(f"\nSugerencia -> TEXTO: {tc}")
    print(f"Sugerencia -> ESTRUCTURADAS: {sc}")

    txt = input(f"\nTEXTO (coma-separado) [{','.join(tc)}]: ").strip() or ",".join(tc)
    text_cols = [c.strip() for c in txt.split(",") if c.strip() in df.columns]

    stx = input(f"ESTRUCTURADAS (coma-separado) [{','.join(sc)}]: ").strip() or ",".join(sc)
    structured_cols = [c.strip() for c in stx.split(",") if c.strip() in df.columns]

    kinds = []
    for col in structured_cols:
        guess = ("dni" if re.search("dni", col, re.I) else
                 "nhc" if re.search("nhc|historia|id", col, re.I) else
                 "phone" if re.search("tel|movil", col, re.I) else
                 "email" if re.search("mail|correo", col, re.I) else
                 "date" if re.search("fecha|hora", col, re.I) else
                 "address" if re.search("direcci|calle", col, re.I) else
                 "name" if re.search("nombre", col, re.I) else "keep")
        k = input(f"Tipo para '{col}' [enter={guess}]: ").strip() or guess
        kinds.append({"column": col, "kind": k})

    profile = {
        "version": 1,
        "model_path": default_model,
        "text_cols": text_cols,
        "structured_cols": kinds,
        "structured_types": {item["column"]: item["kind"] for item in kinds},

        "masking": {
            "strategy": "tag",
            "tag_map": {
                "PER": "[NOMBRE]",
                "GPE": "[LOC]",
                "LOC": "[LOC]"
            }
        },

        "regex": {
            "dni": r"\b\d{8}[A-HJ-NP-TV-Z]\b",
            "nie": r"\b[XYZ]\d{7}[A-HJ-NP-TV-Z]\b",
            "phone": r"\b(?:\+34[\s\-]?)?(?:\d[\s\-]?){9}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}\s+(?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)\.?\s+\d{2,4}\b",
            "nhc": ""
        },

        "gazetteer_cities": [],
        "gazetteer_files": [],
        "ner_labels": ["PER", "GPE", "LOC"]
    }

    Path("profiles").mkdir(exist_ok=True, parents=True)
    outp = Path("profiles") / f"{profile_name}.yaml"
    with open(outp, "w", encoding="utf-8") as f:
        yaml.safe_dump(profile, f, sort_keys=False, allow_unicode=True)

    print(f"\nPerfil guardado: {outp}")
    return str(outp)
# ============================
# EJECUCIÓN CON PERFIL + AUDITORÍA
# ============================
def run_excel_with_profile(xlsx_path: str, profile_path: str, out_path: str = None,
                           dry_run: bool = False, preview: int = 0):
    with open(profile_path, "r", encoding="utf-8") as f:
        prof_raw = yaml.safe_load(f)

    P = normalize_profile(prof_raw)
    df = pd.read_excel(xlsx_path)

    # Cargar modelo principal
    nlp_main = load_main_nlp(P["model_path"])

    tag_map = P["tag_map"]
    regex_cfg = P["regex_map"]
    strat = P["masking_strategy"]
    ner_labels = set(P["ner_labels"]) & ALLOWED_MAIN_LABELS  # reforzar conservador
    gazetteer = P["gazetteer"]

    audit = {"per_col": {}, "per_ent": {}, "rows": int(len(df))}
    def _inc(d, k, n=1): d[k] = int(d.get(k, 0) + n)

    # 1) TEXTO en lote
    for col in P["text_cols"]:
        if col not in df.columns:
            print(f"[AVISO] Columna de texto no encontrada: {col}")
            continue
        col_vals = df[col].astype(str).fillna("").tolist()
        anon_vals, cnt = anonymize_texts(
            col_vals,
            nlp_main=nlp_main,
            tag_map=tag_map,
            regex_cfg=regex_cfg,
            ner_labels=ner_labels,
            gazetteer=gazetteer,
            masking_strategy=strat
        )
        df[col] = anon_vals
        for k, v in (cnt.get("ner") or {}).items():
            _inc(audit["per_ent"], k, v)
        _inc(audit["per_col"], col, sum((cnt.get("ner") or {}).values()))

    # 2) ESTRUCTURADAS
    for col in df.columns:
        kind = P["stypes"].get(col, "keep")
        if kind not in ("keep", "num", "cat", "date", "pii", "dni", "nie", "nhc", "phone", "email", "address", "name"):
            continue
        total = 0
        if kind in ("keep", "num", "cat"):
            pass
        else:
            new_vals = []
            for val in df[col]:
                anon, n = anonymize_struct_value(val, kind, regex_cfg, strat)
                new_vals.append(anon)
                total += n
            df[col] = new_vals
        _inc(audit["per_col"], col, total)

    # Auditoría
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rundir = Path("audit") / f"run_{ts}"
    rundir.mkdir(parents=True, exist_ok=True)

    norm_profile_dump = {
        "text_cols": P["text_cols"],
        "structured_types": P["stypes"],
        "ner_labels": list(ner_labels),
        "regex": P["regex_map"],
        "gazetteer_size": len(P["gazetteer"]),
        "masking": {"strategy": P["masking_strategy"], "tag_map": P["tag_map"]},
        "secondary_ner_loaded": False
    }
    with open(rundir / "normalized_profile.json", "w", encoding="utf-8") as f:
        json.dump(norm_profile_dump, f, ensure_ascii=False, indent=2)
    shutil.copy(profile_path, rundir / "profile.yaml")

    meta = getattr(nlp_main, "meta", {})
    with open(rundir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if preview > 0:
        df.head(preview).to_excel(rundir / "preview_sample.xlsx", index=False)

    if not dry_run:
        outp = out_path or (Path(xlsx_path).with_name(Path(xlsx_path).stem + "_anon.xlsx"))
        Path(outp).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(outp, index=False)
        print(f"\n✔ Excel anonimizado: {outp}")

    print(f"✔ Auditoría: {rundir}")

# ============================
# CLI
# ============================
def main():
    ap = argparse.ArgumentParser(description="ANONIM Excel (conservador, rápido, sin secundario)")
    ap.add_argument("--excel", required=True, help="Ruta del .xlsx de entrada")
    ap.add_argument("--wizard", action="store_true", help="Lanza asistente para crear perfil")
    ap.add_argument("--profile-name", default="perfil", help="Nombre del perfil a crear (sin ruta)")
    ap.add_argument("--profile", help="Ruta a profiles/<nombre>.yaml (obligatoria si no usas --wizard)")
    ap.add_argument("--out", help="Ruta .xlsx de salida")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--preview", type=int, default=0)
    args = ap.parse_args()

    xlsx = args.excel

    if args.wizard:
        prof_path = cli_wizard_build_profile(xlsx, args.profile_name)
        print(f"\nUsa luego:\n  python anonim_meddocan_real_1.py --excel \"{xlsx}\" --profile \"{prof_path}\" --preview 10")
        return

    if not args.profile:
        raise SystemExit("--profile es obligatorio si no usas --wizard")

    run_excel_with_profile(
        xlsx_path=xlsx,
        profile_path=args.profile,
        out_path=args.out,
        dry_run=args.dry_run,
        preview=args.preview
    )

if __name__ == "__main__":
    main()
