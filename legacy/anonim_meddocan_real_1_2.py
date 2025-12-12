# -*- coding: utf-8 -*-
"""
ANONIM: Modo Excel con mapeo dinámico + auditoría
- Wizard para crear perfil YAML (mapeo columnas + regex + etiquetas)
- Procesamiento Excel: columnas de texto libre via spaCy NER + regex
  y columnas estructuradas via regex (o enmascarado completo)
- Auditoría: carpeta audit/run_YYYYMMDD_HHMMSS con perfil, meta, resumen, preview

Requisitos:
  pip install spacy==3.8.7 pandas pyyaml openpyxl
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
except Exception as e:
    raise SystemExit("Falta PyYAML. Instala con: pip install pyyaml")

try:
    import spacy
except Exception as e:
    raise SystemExit("Falta spaCy. Instala con: pip install spacy==3.8.7")

# ----------------------------
# Carga de modelo spaCy (con caché)
# ----------------------------
_NLP = None

def load_nlp(model_path: str):
    global _NLP
    if _NLP is None:
        model_path = str(Path(model_path))
        _NLP = spacy.load(model_path)
    return _NLP

# ----------------------------
# Anonimización de texto libre (NER + regex)
# ----------------------------
def anonymize_text(text: str, nlp, tag_map: dict, regex_cfg: dict, masking_strategy="tag"):
    """
    Devuelve (texto_anonimizado, conteos_dict)
    - Aplica NER para sustituir spans por etiquetas [TIPO] (o █ con redact)
    - Después aplica regex (dni, nie, phone, email, date, nhc) como salvaguarda
    """
    if not isinstance(text, str) or not text.strip():
        return text, {"ner": {}, "regex": {}}

    doc = nlp(text)

    # Reemplazar spans NER del final al inicio para no romper offsets
    spans = sorted([(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents],
                   key=lambda x: x[0], reverse=True)

    ner_counts = {}
    buf = text
    for start, end, label in spans:
        repl = tag_map.get(label, f"[{label}]") if masking_strategy == "tag" else "█" * max(1, end - start)
        buf = buf[:start] + repl + buf[end:]
        ner_counts[label] = ner_counts.get(label, 0) + 1

    # Post-proceso regex
    rx_counts = {}

    def _apply(pattern, tag):
        nonlocal buf, rx_counts
        if not pattern:
            return
        buf, n = re.subn(pattern, tag, buf, flags=re.IGNORECASE)
        if n:
            rx_counts[tag] = rx_counts.get(tag, 0) + n

    _apply(regex_cfg.get("dni"), "[DNI]")
    _apply(regex_cfg.get("nie"), "[NIE]")
    _apply(regex_cfg.get("phone"), "[TEL]")
    _apply(regex_cfg.get("email"), "[EMAIL]")
    _apply(regex_cfg.get("date"), "[FECHA]")
    if regex_cfg.get("nhc"):
        _apply(regex_cfg.get("nhc"), "[ID]")

    return buf, {"ner": ner_counts, "regex": rx_counts}

# ----------------------------
# Anonimización de columnas estructuradas (regex o enmascarado completo)
# ----------------------------
def anonymize_struct_value(val: str, kind: str, regex_cfg: dict, masking_strategy="tag"):
    """
    Para valores de columnas estructuradas (DNI/NHC/phone/email/fecha/address/name),
    intenta regex por tipo; si no hay patrón o no casa, enmascara todo por seguridad.
    """
    if not isinstance(val, str) or not val:
        return val, 0

    tag_by_kind = {
        "dni": "[DNI]", "nie": "[NIE]", "nhc": "[ID]", "phone": "[TEL]",
        "email": "[EMAIL]", "date": "[FECHA]", "address": "[CALLE]", "name": "[NOMBRE]"
    }
    tag = tag_by_kind.get(kind, "[PII]")

    # Para "name" en estructurada, enmascaramos el valor completo
    if kind == "name":
        return (tag if masking_strategy == "tag" else "█" * len(val)), 1

    pattern = regex_cfg.get(kind)
    if not pattern:
        # Sin patrón -> enmascara todo como salvaguarda
        return (tag if masking_strategy == "tag" else "█" * len(val)), 1

    new, n = re.subn(pattern, tag, val, flags=re.IGNORECASE)
    if n == 0:
        # Si no hubo match pero consideramos sensible, enmasCara completo
        return (tag if masking_strategy == "tag" else "█" * len(val)), 1
    return new, n

# ----------------------------
# Heurísticas para sugerir mapeo
# ----------------------------
HEUR_TEXT = re.compile(r"(texto|nota|anamnesis|evoluci[oó]n|informe|descrip)", re.I)
HEUR_STRUCT = re.compile(r"(dni|nie|nhc|historia|id|tel[eé]fono|tel|m[oó]vil|email|correo|direcci[oó]n|calle|pais|cp|c[oó]digo_postal|fecha)", re.I)

def suggest_mapping(df: pd.DataFrame):
    text_cols, struct_cols = [], []
    for c in df.columns:
        sample = df[c].dropna().astype(str).head(50)
        avg_len = (sample.map(len).mean() if not sample.empty else 0)
        if HEUR_TEXT.search(c) or avg_len > 150:
            text_cols.append(c)
        elif HEUR_STRUCT.search(c) or avg_len < 40:
            struct_cols.append(c)
    # Garantizar al menos 'texto' si existe
    if "texto" in df.columns and "texto" not in text_cols:
        text_cols = ["texto"] + text_cols
    return text_cols, struct_cols

# ----------------------------
# Wizard (genera perfil .yaml)
# ----------------------------
def cli_wizard_build_profile(xlsx_path: str, profile_name: str, default_model="models/ner_meddocan/model-best"):
    """
    Lee Excel, sugiere mapeo y genera profiles/<profile_name>.yaml
    """
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
                 "date" if re.search("fecha", col, re.I) else
                 "address" if re.search("direccion|calle", col, re.I) else
                 "name" if re.search("nombre", col, re.I) else "pii")
        k = input(f"Tipo para '{col}' [enter={guess}]: ").strip() or guess
        kinds.append({"column": col, "kind": k})

    profile = {
        "version": 1,
        "model_path": default_model,
        "text_cols": text_cols,
        "structured_cols": kinds,
        "masking": {
            "strategy": "tag",
            "tag_map": {
                "NOMBRE_SUJETO_ASISTENCIA": "[NOMBRE]",
                "ID_SUJETO_ASISTENCIA": "[ID]",
                "NUMERO_TELEFONO": "[TEL]",
                "CORREO_ELECTRONICO": "[EMAIL]",
                "FECHAS": "[FECHA]",
                "HOSPITAL": "[HOSPITAL]",
                "CALLE": "[CALLE]",
                "TERRITORIO": "[TERRITORIO]",
                "PAIS": "[PAIS]"
            }
        },
        "regex": {
            "dni": r"\b\d{8}[A-HJ-NP-TV-Z]\b",
            "nie": r"\b[XYZ]\d{7}[A-HJ-NP-TV-Z]\b",
            "phone": r"\b(?:\+34\s?)?(?:\d\s?){9}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}\s+(?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)\.?\s+\d{2,4}\b",
            "nhc": ""  # Rellena con el patrón del hospital si existe
        },
        "audit": {"keep_input_columns": True, "add_counts_per_row": True}
    }

    Path("profiles").mkdir(exist_ok=True, parents=True)
    outp = Path("profiles") / f"{profile_name}.yaml"
    with open(outp, "w", encoding="utf-8") as f:
        yaml.safe_dump(profile, f, sort_keys=False, allow_unicode=True)
    print(f"\nPerfil guardado: {outp}")
    return str(outp)

# ----------------------------
# Ejecución con perfil + auditoría
# ----------------------------
def run_excel_with_profile(xlsx_path: str, profile_path: str, out_path: str = None,
                           dry_run: bool = False, preview: int = 0):
    with open(profile_path, "r", encoding="utf-8") as f:
        prof = yaml.safe_load(f)

    df = pd.read_excel(xlsx_path)

    nlp = load_nlp(prof["model_path"])
    tag_map = prof["masking"]["tag_map"]
    regex_cfg = prof["regex"]
    strat = prof["masking"]["strategy"]

    audit = {"per_col": {}, "per_ent": {}, "rows": int(len(df))}
    def _inc(d, k, n=1): d[k] = int(d.get(k, 0) + n)

    # Procesar columnas de TEXTO (NER + regex)
    for col in prof["text_cols"]:
        if col not in df.columns:
            print(f"[AVISO] Columna de texto no encontrada: {col}")
            continue
        new_vals, ner_counts_total, rx_counts_total = [], {}, {}
        for val in df[col].astype(str).fillna(""):
            anon, counts = anonymize_text(val, nlp, tag_map, regex_cfg, strat)
            new_vals.append(anon)
            for k, v in counts["ner"].items(): _inc(ner_counts_total, k, v)
            for k, v in counts["regex"].items(): _inc(rx_counts_total, k, v)
        df[col] = new_vals
        # actualizar auditoría
        for k, v in ner_counts_total.items(): _inc(audit["per_ent"], k, v)
        _inc(audit["per_col"], col, sum(ner_counts_total.values()) + sum(rx_counts_total.values()))

    # Procesar columnas ESTRUCTURADAS (regex/enmascarado)
    for item in prof["structured_cols"]:
        col, kind = item["column"], item["kind"]
        if col not in df.columns:
            print(f"[AVISO] Columna estructurada no encontrada: {col}")
            continue
        new_vals, total = [], 0
        for val in df[col].astype(str).fillna(""):
            anon, n = anonymize_struct_value(val, kind, regex_cfg, strat)
            new_vals.append(anon)
            total += n
        df[col] = new_vals
        _inc(audit["per_col"], col, total)

    # Auditoría
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rundir = Path("audit") / f"run_{ts}"
    rundir.mkdir(parents=True, exist_ok=True)
    shutil.copy(profile_path, rundir / "profile.yaml")

    # meta del modelo
    meta = getattr(nlp, "meta", {})
    with open(rundir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(rundir / "columns_map.json", "w", encoding="utf-8") as f:
        json.dump({"text_cols": prof["text_cols"], "structured_cols": prof["structured_cols"]},
                  f, ensure_ascii=False, indent=2)

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

# ----------------------------
# CLI
# ----------------------------
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
