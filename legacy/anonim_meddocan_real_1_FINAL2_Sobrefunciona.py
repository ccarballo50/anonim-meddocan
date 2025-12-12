# -*- coding: utf-8 -*-
"""
ANONIM: Excel con mapeo dinámico + auditoría (versión con NER secundario para ubicaciones/organizaciones)
- Wizard para crear perfil YAML (mapeo columnas + regex + etiquetas)
- Texto libre: spaCy NER (modelo principal) + spaCy NER general (es_core_news_md/lg) + regex + gazetteer + heurística
- Estructuradas: keep/num/cat (se conservan), date->año, pii->[X], tipos específicos por regex (dni/nie/nhc/phone/email/address/name)
- Auditoría: audit/run_YYYYMMDD_HHMMSS con perfil, meta, resumen y preview

Requisitos mínimos:
  pip install spacy==3.8.7 pandas pyyaml openpyxl
  (Opcional, recomendado) pip install es_core_news_md
    - o alternativamente instalar 'es_core_news_lg'
"""

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

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
# Carga de modelos (caché)
# ============================
_NLP_MAIN = None
_NLP_SEC = None  # es_core_news_md/lg para ubicaciones/organizaciones

def load_main_nlp(model_path: str):
    """Carga el modelo principal (MedDOCAN entrenado)"""
    global _NLP_MAIN
    if _NLP_MAIN is None:
        _NLP_MAIN = spacy.load(str(Path(model_path)))
    return _NLP_MAIN

def load_secondary_nlp():
    """Carga el modelo general español para detectar LOC/GPE/ORG/etc. (si existe)"""
    global _NLP_SEC
    if _NLP_SEC is not None:
        return _NLP_SEC
    # Intentar md, luego lg
    for pkg in ("es_core_news_md", "es_core_news_lg"):
        try:
            _NLP_SEC = spacy.load(pkg)
            return _NLP_SEC
        except Exception:
            continue
    # Si no hay modelo general instalado, no pasa nada: se trabaja solo con el principal
    _NLP_SEC = None
    return None


# ============================
# Utilidades de perfil
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
            # Ignorar ficheros no accesibles
            pass
    return terms

def normalize_profile(profile: dict):
    """
    Devuelve diccionario normalizado:
      - text_cols: lista de columnas de texto
      - stypes: dict {col: kind}
      - ner_labels: lista de etiquetas NER a anonimizar (para ambos modelos)
      - regex_map: dict de regex complementarias
      - gazetteer: set de términos (ciudades/centros) a ofuscar como [LOC]
      - masking_strategy: 'tag' (por defecto) o 'redact'
      - tag_map: dict para mapear etiquetas del modelo a tokens
      - model_path: ruta del modelo principal spaCy
    """
    # Columnas de TEXTO
    text_cols = _get_first(profile, "text_columns", "text_cols", default=[]) or []

    # Tipos estructurados (dict o lista)
    stypes = _get_first(profile, "structured_types", default={}) or {}
    if not stypes:
        scols = _get_first(profile, "structured_cols", default=[]) or []
        if isinstance(scols, list):
            stypes = {
                item["column"]: item["kind"]
                for item in scols
                if isinstance(item, dict) and "column" in item and "kind" in item
            }

    # Etiquetas NER a anonimizar (se aplican a ambos modelos)
    ner_labels = _get_first(profile, "ner_labels", default=[
        "PER", "ORG", "LOC", "GPE", "NORP", "FAC", "MISC"
    ])

    # Regex adicionales
    regex_map = _get_first(profile, "regex", default={}) or {}

    # Gazetteer (YAML + ficheros)
    gaz_cities = set(_get_first(profile, "gazetteer_cities", default=[]) or [])
    gaz_files = _get_first(profile, "gazetteer_files", default=[]) or []
    gaz_from_files = _load_gazetteer_from_files(gaz_files)
    gazetteer = gaz_cities.union(gaz_from_files)

    # Enmascarado
    masking = _get_first(profile, "masking", default={}) or {}
    masking_strategy = masking.get("strategy", "tag")
    tag_map = masking.get("tag_map", {}) or {}

    # Modelo principal
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
# Reemplazo de spans seguro
# ============================
def replace_spans(text, spans, token="[X]"):
    """
    Reemplaza spans (lista de (start, end)) por token, uniendo solapes
    """
    if not spans:
        return text
    spans = sorted(spans)
    merged = []
    s0, e0 = spans[0]
    for s, e in spans[1:]:
        if s <= e0:
            e0 = max(e0, e)
        else:
            merged.append((s0, e0))
            s0, e0 = s, e
    merged.append((s0, e0))

    out = []
    last = 0
    for s, e in merged:
        out.append(text[last:s])
        out.append(token)
        last = e
    out.append(text[last:])
    return "".join(out)


# ============================
# Anonimización de texto libre
# ============================
def anonymize_text(text: str, nlp_main, nlp_sec, tag_map: dict, regex_cfg: dict,
                   ner_labels, gazetteer, masking_strategy="tag"):
    """
    Devuelve (texto_anonimizado, conteos_dict)
    Pipeline:
      1) NER principal (MedDOCAN u otro)
      2) NER secundario (es_core_news_md/lg) para LOC/GPE/ORG/PER/FAC/NORP
      3) Regex complementarias (dni/nie/phone/email/date/nhc)
      4) Gazetteer de ciudades/centros -> [LOC]
      5) Heurística "vive/reside/domiciliad* en <Topónimo>" -> [LOC]
    """
    if not isinstance(text, str) or not text.strip():
        return text, {"ner": {}, "regex": {}, "gazetteer": 0, "heur": 0, "ner2": {}}

    buf = text

    ner_counts = {}
    ner2_counts = {}

    # 1) NER principal
    try:
        doc = nlp_main(buf)
        spans_main = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents if ent.label_ in ner_labels]
    except Exception:
        spans_main = []

    # 2) NER secundario (si existe)
    spans_sec = []
    if nlp_sec is not None:
        try:
            doc2 = nlp_sec(buf)
            spans_sec = [(ent.start_char, ent.end_char, ent.label_) for ent in doc2.ents if ent.label_ in ner_labels]
        except Exception:
            spans_sec = []

    # Fusionar spans de ambos modelos y reemplazar del final al inicio
    # Conservamos la etiqueta original para el tag_map.
    all_spans = sorted(spans_main + spans_sec, key=lambda x: x[0], reverse=True)
    for s, e, lab in all_spans:
        repl = tag_map.get(lab, f"[{lab}]") if masking_strategy == "tag" else "█"
        buf = buf[:s] + repl + buf[e:]
        # Contabilizar por procedencia (aprox): lab en ner_counts (principal), ner2_counts (secundario)
        # No diferenciamos exactamente el origen por simplicidad; si quisieras, deberíamos marcarlo antes.
        ner_counts[lab] = ner_counts.get(lab, 0) + 1

    # 3) Regex complementarias
    rx_counts = {}
    def _apply(pattern, tag_token):
        nonlocal buf, rx_counts
        if not pattern:
            return
        new, n = re.subn(pattern, tag_token, buf, flags=re.IGNORECASE)
        if n:
            buf = new
            rx_counts[tag_token] = int(rx_counts.get(tag_token, 0) + n)

    _apply(regex_cfg.get("dni"), "[DNI]")
    _apply(regex_cfg.get("nie"), "[NIE]")
    _apply(regex_cfg.get("phone"), "[TEL]")
    _apply(regex_cfg.get("email"), "[EMAIL]")
    _apply(regex_cfg.get("date"), "[FECHA]")
    if regex_cfg.get("nhc"):
        _apply(regex_cfg.get("nhc"), "[ID]")

    # 4) Gazetteer -> [LOC]
    gaz_count = 0
    if gazetteer:
        pat = r"\b(" + "|".join(re.escape(t) for t in gazetteer if t) + r")\b"
        new, n = re.subn(pat, "[LOC]", buf, flags=re.IGNORECASE)
        if n:
            buf = new
            gaz_count += n

    # 5) Heurística "vive/reside/domiciliad* en <Topónimo>"
    heur_pat = r"\b(vive en|reside en|domiciliad[oa] en)\s+[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑ\-]+"
    new, n = re.subn(heur_pat, r"\1 [LOC]", buf, flags=re.IGNORECASE)
    heur_count = n if n else 0
    if n:
        buf = new

    return buf, {"ner": ner_counts, "ner2": ner2_counts, "regex": rx_counts, "gazetteer": gaz_count, "heur": heur_count}


# ============================
# Estructuradas
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

    # Conservar
    if kind in ("num", "cat", "keep"):
        return val, 0

    # Fecha -> año
    if kind == "date":
        return anonymize_date(val), 1

    # PII genérico
    if kind == "pii":
        return mask_full(val, strategy=masking_strategy, token="[X]"), 1

    # Tipos específicos por regex
    tag_by_kind = {
        "dni": "[DNI]", "nie": "[NIE]", "nhc": "[ID]",
        "phone": "[TEL]", "email": "[EMAIL]",
        "address": "[CALLE]", "name": "[NOMBRE]"
    }
    tag = tag_by_kind.get(kind, "[X]")

    if kind == "name":
        return mask_full(val, strategy=masking_strategy, token=tag), 1

    pattern = regex_cfg.get(kind)
    s = str(val)
    if not pattern:
        return mask_full(s, strategy=masking_strategy, token="[X]"), 1

    new, n = re.subn(pattern, tag, s, flags=re.IGNORECASE)
    if n == 0:
        return mask_full(s, strategy=masking_strategy, token="[X]"), 1
    return new, n


# ============================
# Heurísticas para sugerir mapeo
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
    if "texto" in df.columns and "texto" not in text_cols:
        text_cols = ["texto"] + text_cols
    return text_cols, struct_cols


# ============================
# Wizard (creación de perfil)
# ============================
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
                 "address" if re.search("direccion|calle", col, re.I) else
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
                # Si no hay mapeo aquí, se usa [LABEL] por defecto
                "PER": "[NOMBRE]",
                "ORG": "[ORG]",
                "LOC": "[LOC]",
                "GPE": "[LOC]",
                "NORP": "[COLECTIVO]",
                "FAC": "[INSTALACION]",
                "MISC": "[X]"
            }
        },

        "regex": {
            "dni": r"\b\d{8}[A-HJ-NP-TV-Z]\b",
            "nie": r"\b[XYZ]\d{7}[A-HJ-NP-TV-Z]\b",
            "phone": r"\b(?:\+34[\s\-]?)?(?:\d[\s\-]?){9}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}\s+(?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)\.?\s+\d{2,4}\b",
            "nhc": ""  # Si existe patrón concreto del hospital, ponlo aquí
        },

        "audit": {"keep_input_columns": True, "add_counts_per_row": True},

        # Gazetteer opcional (se puede cargar también desde ficheros)
        "gazetteer_cities": [],
        "gazetteer_files": [],

        "ner_labels": ["PER", "ORG", "LOC", "GPE", "NORP", "FAC", "MISC"]
    }

    Path("profiles").mkdir(exist_ok=True, parents=True)
    outp = Path("profiles") / f"{profile_name}.yaml"
    with open(outp, "w", encoding="utf-8") as f:
        yaml.safe_dump(profile, f, sort_keys=False, allow_unicode=True)
    print(f"\nPerfil guardado: {outp}")
    return str(outp)


# ============================
# Ejecución con perfil + auditoría
# ============================
def run_excel_with_profile(xlsx_path: str, profile_path: str, out_path: str = None,
                           dry_run: bool = False, preview: int = 0):
    with open(profile_path, "r", encoding="utf-8") as f:
        prof_raw = yaml.safe_load(f)

    P = normalize_profile(prof_raw)

    df = pd.read_excel(xlsx_path)

    nlp_main = load_main_nlp(P["model_path"])
    nlp_sec = load_secondary_nlp()  # puede ser None si no está instalado

    tag_map = P["tag_map"]
    regex_cfg = P["regex_map"]
    strat = P["masking_strategy"]
    ner_labels = P["ner_labels"]
    gazetteer = P["gazetteer"]

    audit = {"per_col": {}, "per_ent": {}, "rows": int(len(df))}
    def _inc(d, k, n=1): d[k] = int(d.get(k, 0) + n)

    # 1) TEXTO
    for col in P["text_cols"]:
        if col not in df.columns:
            print(f"[AVISO] Columna de texto no encontrada: {col}")
            continue
        new_vals = []
        ner_tot, rx_tot, gaz_tot, heur_tot = {}, {}, 0, 0
        for val in df[col].astype(str).fillna(""):
            anon, counts = anonymize_text(val, nlp_main, nlp_sec, tag_map, regex_cfg,
                                          ner_labels, gazetteer, strat)
            new_vals.append(anon)
            for k, v in counts["ner"].items(): _inc(ner_tot, k, v)
            for k, v in counts["regex"].items(): _inc(rx_tot, k, v)
            gaz_tot += counts["gazetteer"]
            heur_tot += counts["heur"]
        df[col] = new_vals
        for k, v in ner_tot.items(): _inc(audit["per_ent"], k, v)
        _inc(audit["per_col"], col, sum(ner_tot.values()) + sum(rx_tot.values()) + gaz_tot + heur_tot)

    # 2) ESTRUCTURADAS
    #    Por defecto conservar (keep); aplicar reglas solo si se indica
    for col in df.columns:
        kind = P["stypes"].get(col, "keep")
        if kind not in ("keep", "num", "cat", "date", "pii", "dni", "nie", "nhc", "phone", "email", "address", "name"):
            # tipos desconocidos -> conservar
            continue
        total = 0
        if kind in ("keep", "num", "cat"):
            pass  # se conserva
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
        "ner_labels": P["ner_labels"],
        "regex": P["regex_map"],
        "gazetteer_size": len(P["gazetteer"]),
        "masking": {"strategy": P["masking_strategy"], "tag_map": P["tag_map"]},
        "secondary_ner_loaded": bool(nlp_sec is not None)
    }
    with open(rundir / "normalized_profile.json", "w", encoding="utf-8") as f:
        json.dump(norm_profile_dump, f, ensure_ascii=False, indent=2)
    shutil.copy(profile_path, rundir / "profile.yaml")

    meta = getattr(nlp_main, "meta", {})
    with open(rundir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(rundir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

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
    ap = argparse.ArgumentParser(description="ANONIM Excel (mapeo dinámico + auditoría)")
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

    run_excel_with_profile(xlsx_path=xlsx,
                           profile_path=args.profile,
                           out_path=args.out,
                           dry_run=args.dry_run,
                           preview=args.preview)

if __name__ == "__main__":
    main()
