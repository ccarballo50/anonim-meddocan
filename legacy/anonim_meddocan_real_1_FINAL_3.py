# -*- coding: utf-8 -*-
"""
ANONIM: Excel con mapeo dinámico + auditoría (protección de fármacos)
- Texto libre: spaCy NER (modelo principal) + NER secundario (es_core_news_md/lg)
  + regex + gazetteer + heurística, PERO preservando siempre la medicación.
- Estructuradas: keep/num/cat (se conservan), date->año, pii->[X], tipos específicos por regex.
- Auditoría: audit/run_YYYYMMDD_HHMMSS con perfil, meta, resumen y preview.

Requisitos:
  pip install spacy==3.8.7 pandas pyyaml openpyxl
  (Recomendado) python -m spacy download es_core_news_md
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

import joblib

_DRUG_CLF = None
_DRUG_LEX = set()

def load_drug_clf(path="profiles/models/drug_token_clf.joblib"):
    """
    Carga el clasificador de tokens (DRUG/OTHER/etc.) desde joblib.
    """
    global _DRUG_CLF, _DRUG_LEX
    if _DRUG_CLF is not None:
        return _DRUG_CLF
    p = Path(path)
    if p.exists():
        obj = joblib.load(p)
        _DRUG_CLF = obj["pipe"]
        _DRUG_LEX = set(obj.get("lex", []))
    else:
        _DRUG_CLF = None
        _DRUG_LEX = set()
    return _DRUG_CLF

##########################################################
# === Token features para clasificación tipo DRUG SÍ/NO ===
##########################################################

UNITS = {"mg", "mcg", "µg", "ug", "g", "ui", "u.i.", "ml", "mL", "cc"}
TOKEN_RE = re.compile(r"\w+([\-\.]\w+)*", re.UNICODE)

def _char_ngram_feats(tok: str, n=(3,4,5)):
    """Extrae n-gramas de caracteres para el token."""
    s = tok.lower()
    feats = {}
    for k in n:
        for i in range(len(s)-k+1):
            feats[f"char{k}={s[i:i+k]}"] = 1
    return feats

def _ctx_feats(tokens, i: int):
    """Rasgos contextuales: tokens vecinos, unidades de dosis, etc."""
    f = {}
    prev1 = tokens[i-1].lower() if i-1 >= 0 else ""
    next1 = tokens[i+1].lower() if i+1 < len(tokens) else ""
    prev2 = tokens[i-2].lower() if i-2 >= 0 else ""
    next2 = tokens[i+2].lower() if i+2 < len(tokens) else ""
    f["prev1"] = prev1
    f["next1"] = next1
    f["prev2"] = prev2
    f["next2"] = next2
    f["ctx_has_unit"] = int(next1 in UNITS or prev1 in UNITS or next2 in UNITS or prev2 in UNITS)
    f["ctx_has_number"] = int(bool(re.search(r"\d", next1) or re.search(r"\d", prev1)))
    f["is_capitalized"] = int(tokens[i][:1].isupper())
    return f

def _featurize_window(tokens, i: int):
    """Combina n-gramas de caracteres, contexto y lexicón para un token dado."""
    f = {}
    tok = tokens[i]
    f.update(_char_ngram_feats(tok))
    f.update(_ctx_feats(tokens, i))
    f["in_drug_lex"] = int(tok.lower() in _DRUG_LEX)
    return f


# ============================
# Modelos (caché)
# ============================
_NLP_MAIN = None
_NLP_SEC = None  # es_core_news_md/lg

def load_main_nlp(model_path: str):
    global _NLP_MAIN
    if _NLP_MAIN is None:
        _NLP_MAIN = spacy.load(str(Path(model_path)), disable=["tagger","parser","attribute_ruler","lemmatizer","morphologizer"])
    return _NLP_MAIN

def load_secondary_nlp():
    global _NLP_SEC
    if _NLP_SEC is not None:
        return _NLP_SEC
    for pkg in ("es_core_news_md", "es_core_news_lg"):
        try:
            _NLP_SEC = spacy.load(pkg, disable=["tagger","parser","attribute_ruler","lemmatizer","morphologizer"])
            return _NLP_SEC
        except Exception:
            continue
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

def _load_list_file(path: str):
    terms = set()
    if not path:
        return terms
    p = Path(path)
    if not p.exists():
        return terms
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    terms.add(t)
    except Exception:
        pass
    return terms

def _load_gazetteer_from_files(paths):
    terms = set()
    for p in (paths or []):
        terms |= _load_list_file(p)
    return terms

def normalize_profile(profile: dict, profile_path: str = ""):
    """
    Devuelve dict normalizado:
      - text_cols: lista de columnas de texto
      - stypes: dict {col: kind}
      - ner_labels: etiquetas a anonimizar
      - regex_map: dict de regex
      - gazetteer: set términos LOC (opcional)
      - masking_strategy: 'tag'|'redact'
      - tag_map: mapeo etiqueta->token
      - model_path: ruta del modelo principal
      - drug_lexicon: set de fármacos (lista blanca)
    """
    # Columnas de TEXTO
    text_cols = _get_first(profile, "text_columns", "text_cols", default=[]) or []

    # Tipos estructurados
    stypes = _get_first(profile, "structured_types", default={}) or {}
    if not stypes:
        scols = _get_first(profile, "structured_cols", default=[]) or []
        if isinstance(scols, list):
            stypes = {
                item["column"]: item["kind"]
                for item in scols
                if isinstance(item, dict) and "column" in item and "kind" in item
            }

    # Etiquetas NER
    ner_labels = _get_first(profile, "ner_labels", default=[
        "PER", "ORG", "LOC", "GPE", "NORP", "FAC", "MISC"
    ])

    # Regex
    regex_map = _get_first(profile, "regex", default={}) or {}

    # Gazetteer
    gaz_cities = set(_get_first(profile, "gazetteer_cities", default=[]) or [])
    gaz_files = _get_first(profile, "gazetteer_files", default=[]) or []
    gazetteer = gaz_cities.union(_load_gazetteer_from_files(gaz_files))

    # Enmascarado
    masking = _get_first(profile, "masking", default={}) or {}
    masking_strategy = masking.get("strategy", "tag")
    tag_map = masking.get("tag_map", {}) or {}

    # Modelo principal
    model_path = profile.get("model_path", "models/ner_meddocan/model-best")

    # DRUG LEXICON (lista blanca)
    # - drug_lexicon_file: ruta (absoluta o relativa al perfil) a un .txt (1 fármaco por línea)
    # - drug_lexicon_extra: lista inline en YAML
    drug_lexicon = set()
    base_dir = Path(profile_path).parent if profile_path else Path.cwd()
    lex_file = profile.get("drug_lexicon_file", "")
    if lex_file:
        # Permitir ruta relativa al YAML
        lpath = (base_dir / lex_file) if not Path(lex_file).is_absolute() else Path(lex_file)
        drug_lexicon |= _load_list_file(str(lpath))
    # Por compatibilidad: si existe profiles/drug_lexicon.txt, cargarlo
    default_lex = base_dir / "drug_lexicon.txt"
    if default_lex.exists():
        drug_lexicon |= _load_list_file(str(default_lex))
    # Inline
    drug_lexicon |= set(_get_first(profile, "drug_lexicon_extra", default=[]) or [])

    # Normalizar a minúsculas
    drug_lexicon = {t.strip().lower() for t in drug_lexicon if t.strip()}

    return {
        "text_cols": text_cols,
        "stypes": stypes,
        "ner_labels": ner_labels,
        "regex_map": regex_map,
        "gazetteer": gazetteer,
        "masking_strategy": masking_strategy,
        "tag_map": tag_map,
        "model_path": model_path,
        "drug_lexicon": drug_lexicon
    }


# ============================
# Reemplazo de spans y protección
# ============================
def replace_spans(text, spans, token="[X]"):
    """Reemplaza spans (lista de (start, end)) por token, uniendo solapes."""
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

    out, last = [], 0
    for s, e in merged:
        out.append(text[last:s])
        out.append(token)
        last = e
    out.append(text[last:])
    return "".join(out)

def spans_overlap(a, b):
    """True si (s1,e1) solapa con (s2,e2)."""
    s1, e1 = a
    s2, e2 = b
    return not (e1 <= s2 or e2 <= s1)

def build_placeholders(text, spans):
    """
    Congela los spans protegidos sustituyéndolos por placeholders __PROT{i}__.
    Devuelve (texto_con_placeholders, mapping{id->original}, lista_spans_placeholders)
    """
    if not spans:
        return text, {}, []
    # Unir solapes
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

    out, last = [], 0
    mapping = {}
    new_spans = []
    for i, (s, e) in enumerate(merged):
        token = f"__PROT{i}__"
        mapping[token] = text[s:e]
        out.append(text[last:s])
        out.append(token)
        new_spans.append((len("".join(out)) - len(token), len("".join(out))))
        last = e
    out.append(text[last:])
    return "".join(out), mapping, new_spans

def restore_placeholders(text, mapping: dict):
    """Restaura placeholders __PROT{i}__ a su texto original."""
    if not mapping:
        return text
    # Reemplazar tokens por su valor original
    def repl(m):
        tok = m.group(0)
        return mapping.get(tok, tok)
    pattern = r"__PROT\d+__"
    return re.sub(pattern, repl, text)


# ============================
# Protección de fármacos (detección)
# ============================
# === DRUG patterns (super robustos) ===

DRUG_SUFFIXES = [
    "ina","pril","sartan","statina","mab","cumarol","xaban","zepam","zolam",
    "oxetina","ciclina","micina","prazol","dipino","olol","arona","vastatina"
]
DOSE_UNITS = r"(?:mg|mcg|µg|ug|g|ui|ml|cc|%)\.?"

# Contextos farmacológicos: permite ':' o '-' tras 'con' y hasta 3 tokens siguientes
CTX_PAT = re.compile(
    r"(?i)\b("
    r"tratamiento(?:\s+cr[oó]nico)?\s+con|tto\.?\s+con|toma|tomando|tomas|"
    r"al[ée]rgic[oa]\s+a|alergia\s+a|antiagregado\s+con|anticoagulado\s+con|"
    r"profilaxis\s+con|en\s+tratamiento\s+con"
    r")\s*(?:[:\-])?\s+"
    r"([A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑ\-]*(?:\s+[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑ\-]*){0,3})"
)

# Fármaco + dosis CON unidad (Adiro 100 mg, atorvastatina 40mg, SINTROM 2 mg.)
DOSE_PAT = re.compile(
    rf"(?i)\b([A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑ\-]+)\s*(\d+(?:[.,]\d+)?)\s*{DOSE_UNITS}\b"
)

# Fármaco + número SIN unidad (Adiro 100) — muy frecuente
DRUG_NUM_NO_UNIT_PAT = re.compile(
    r"(?i)\b([A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑ\-]+)\s+(\d+(?:[.,]\d+)?)\b"
)



def find_drug_spans(text: str, drug_lexicon: set):
    spans = []
    if not isinstance(text, str) or not text:
        return spans

    # 1) Lexicón (permite ®/™; multi-palabra con guiones/espacios)
    if drug_lexicon:
        terms = sorted({t.strip().lower() for t in drug_lexicon if t.strip()}, key=len, reverse=True)
        for term in terms:
            parts = [re.escape(p) for p in term.split()]
            term_pat = r"(?:\s+|\-)+".join(parts) if len(parts) > 1 else parts[0]
            pat = re.compile(rf"(?i)(?<!\w){term_pat}(?:[®™])?(?!\w)")
            for m in pat.finditer(text):
                spans.append((m.start(), m.end()))

    # 2) Contexto farmacológico
    for m in CTX_PAT.finditer(text):
        spans.append((m.start(2), m.end(2)))   # nombre (hasta 3 tokens)

    # 3) Fármaco + dosis CON unidad
    for m in DOSE_PAT.finditer(text):
        spans.append((m.start(0), m.end(0)))   # toda la expresión "X 100 mg"

    # 4) Fármaco + número SIN unidad
    for m in DRUG_NUM_NO_UNIT_PAT.finditer(text):
        spans.append((m.start(0), m.end(0)))

    # 5) Sufijos típicos + número/unidad cerca (±25 chars)
    WORD_PAT = re.compile(r"(?i)\b([A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑ\-]+)\b")
    for m in WORD_PAT.finditer(text):
        w = m.group(1).lower()
        if any(w.endswith(suf) for suf in DRUG_SUFFIXES):
            s, e = m.start(1), m.end(1)
            ctx = text[max(0, s-25):min(len(text), e+25)]
            if re.search(rf"{DOSE_UNITS}|\d", ctx, flags=re.I):
                spans.append((s, e))

    # Unificar solapes
    if not spans:
        return spans
    spans.sort()
    merged = []
    s0, e0 = spans[0]
    for s, e in spans[1:]:
        if s <= e0:
            e0 = max(e0, e)
        else:
            merged.append((s0, e0))
            s0, e0 = s, e
    merged.append((s0, e0))
    return merged


def is_probably_drug_segment(buf: str, s: int, e: int, drug_lexicon: set) -> bool:
    """
    Devuelve True si el segmento buf[s:e] probablemente sea medicación.
    Se usa como ÚLTIMO CORTAFUEGOS dentro del bucle NER:
      - Si aquí devolvemos True, NO sustituimos por [NOMBRE]/[ORG]/etc.
    """
    seg = buf[s:e]
    if "__PROT" in seg:
        return True

    # ¿Contiene tokens presentes en el lexicón?
    toks = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-]+", seg)
    low = [t.lower() for t in toks]
    if drug_lexicon and any(t in drug_lexicon for t in low):
        return True

    # ¿Parece fármaco + dosis dentro del propio segmento?
    if DOSE_PAT.search(seg) or DRUG_NUM_NO_UNIT_PAT.search(seg):
        return True

    # Contexto a la izquierda: "... tratamiento/toma/antiagregado/anticoagulado ... con <seg>"
    left = buf[max(0, s-40):s]
    if re.search(r"(?i)(tratamiento|tto\.?|toma|tomando|antiagregado|anticoagulado|profilaxis)\s+con\s*$", left):
        return True

    return False


# ============================
# Anonimización de texto (con protección de fármacos)
# ============================
def anonymize_text(text: str, nlp_main, nlp_sec, tag_map: dict, regex_cfg: dict,
                   ner_labels, gazetteer, drug_lexicon: set, masking_strategy="tag"):
    """
    Devuelve (texto_anonimizado, conteos)
    Orden:
      0) Detectar y PROTEGER spans de medicación -> placeholders __PROT{i}__
      1) NER principal
      2) NER secundario
      3) Regex complementarias
      4) Gazetteer [LOC]
      5) Heurística vive/reside/domiciliad* en <Topónimo> -> [LOC]
      6) Restaurar placeholders (medicación intacta)
    """
    if not isinstance(text, str) or not text.strip():
        return text, {"ner": {}, "ner2": {}, "regex": {}, "gazetteer": 0, "heur": 0, "protected_drug_spans": 0}

    buf = text
    # Normalizar espacios raros (NBSP, etc.)
    # Cargar clasificador de fármacos (si existe)
    clf = load_drug_clf()  # usa la ruta por defecto: profiles/models/drug_token_clf.joblib


    # 0) Proteger medicación
    drug_spans = find_drug_spans(buf, drug_lexicon)
    buf, placeholder_map, _ = build_placeholders(buf, drug_spans)
    protected_count = len(placeholder_map)

    ner_counts = {}
    ner2_counts = {}

    # 1) NER principal
    try:
        doc = nlp_main(buf)
        spans_main = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents if ent.label_ in ner_labels]
    except Exception:
        spans_main = []

    # 2) NER secundario
    spans_sec = []
    if nlp_sec is not None:
        try:
            doc2 = nlp_sec(buf)
            spans_sec = [(ent.start_char, ent.end_char, ent.label_) for ent in doc2.ents if ent.label_ in ner_labels]
        except Exception:
            spans_sec = []

    # Fusionar y reemplazar (del final al inicio)
    all_spans = sorted(spans_main + spans_sec, key=lambda x: x[0], reverse=True)

# Reemplazo NER con veto farmacológico
    for s, e, lab in all_spans:
        segment = buf[s:e]  # texto candidato a anonimizar
        do_replace = True

        # Si hay clasificador, analizamos si el span contiene algún token DRUG
        if clf is not None:
            # Tomamos una ventana contextual alrededor del span
            start_win = max(0, s - 40)
            end_win = min(len(buf), e + 40)
            win_text = buf[start_win:end_win]

            # Tokenizamos la ventana
            tokens = [m.group(0) for m in TOKEN_RE.finditer(win_text)]

            # Revisamos los tokens que se solapan con el span
            is_drug = False
            for m in TOKEN_RE.finditer(win_text):
                ts, te = m.span()
                abs_s, abs_e = start_win + ts, start_win + te
                overlap = not (abs_e <= s or abs_s >= e)
                if overlap:
                    # localizamos índice del token
                    idx = len([1 for x in TOKEN_RE.finditer(win_text) if x.start() < ts])
                    feats = _featurize_window(tokens, idx)
                    try:
                        pred = clf.predict([feats])[0]
                    except:
                        pred = "OTHER"

                    if pred.upper() == "DRUG":
                        is_drug = True
                        break

            if is_drug:
                # No reemplazamos, preservamos el fármaco
                do_replace = False

        # Si no es DRUG, se anonimiza como antes
        if do_replace:
            repl = tag_map.get(lab, f"[{lab}]") if masking_strategy == "tag" else "█"
            buf = buf[:s] + repl + buf[e:]
            ner_counts[lab] = ner_counts.get(lab, 0) + 1



    # 3) Regex complementarias (evitar tocar placeholders)
    rx_counts = {}
    def safe_sub(pattern, token):
        nonlocal buf, rx_counts
        if not pattern:
            return
        # Partimos por placeholders y aplicamos regex solo en secciones no protegidas
        parts = re.split(r"(__PROT\d+__)", buf)
        for i in range(0, len(parts), 2):  # solo en pares (texto fuera de placeholders)
            new, n = re.subn(pattern, token, parts[i], flags=re.IGNORECASE)
            if n:
                parts[i] = new
                rx_counts[token] = int(rx_counts.get(token, 0) + n)
        buf = "".join(parts)

    safe_sub(regex_cfg.get("dni"), "[DNI]")
    safe_sub(regex_cfg.get("nie"), "[NIE]")
    safe_sub(regex_cfg.get("phone"), "[TEL]")
    safe_sub(regex_cfg.get("email"), "[EMAIL]")
    safe_sub(regex_cfg.get("date"), "[FECHA]")
    if regex_cfg.get("nhc"):
        safe_sub(regex_cfg.get("nhc"), "[ID]")

    # 4) Gazetteer -> [LOC] (sin tocar placeholders)
    gaz_count = 0
    if gazetteer:
        pat = r"\b(" + "|".join(re.escape(t) for t in gazetteer if t) + r")\b"
        parts = re.split(r"(__PROT\d+__)", buf)
        for i in range(0, len(parts), 2):
            new, n = re.subn(pat, "[LOC]", parts[i], flags=re.IGNORECASE)
            if n:
                parts[i] = new
                gaz_count += n
        buf = "".join(parts)

    # 5) Heurística "vive/reside/domiciliad* en <Topónimo>" (sin tocar placeholders)
    heur_pat = r"\b(vive en|reside en|domiciliad[oa] en)\s+[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑ\-]+"
    parts = re.split(r"(__PROT\d+__)", buf)
    heur_count = 0
    for i in range(0, len(parts), 2):
        new, n = re.subn(heur_pat, r"\1 [LOC]", parts[i], flags=re.IGNORECASE)
        if n:
            parts[i] = new
            heur_count += n
    buf = "".join(parts)

    # 6) Restaurar medicación protegida (NUNCA se anonimiza)
    buf = restore_placeholders(buf, placeholder_map)

    return buf, {
        "ner": ner_counts,
        "ner2": ner2_counts,
        "regex": rx_counts,
        "gazetteer": gaz_count,
        "heur": heur_count,
        "protected_drug_spans": protected_count
    }


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
# Heurísticas de mapeo
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
# Wizard (creación perfil)
# ============================
def cli_wizard_build_profile(xlsx_path: str, profile_name: str, default_model="models/ner_meddocan/model-best"):
    df = p# Lee sólo las primeras N filas cuando es preview (mucho más rápido que leer todo y luego .head)
    if preview and preview > 0:
        df = pd.read_excel(xlsx_path, nrows=preview)
    else:
        df = pd.read_excel(xlsx_path)
    
# --- FAST PREVIEW: si hay preview, solo procesa esas N filas ---

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
            "nhc": ""
        },

        "audit": {"keep_input_columns": True, "add_counts_per_row": True},

        # Gazetteer opcional
        "gazetteer_cities": [],
        "gazetteer_files": [],

        # DRUGS: configuración opcional (si tienes lista externa)
        "drug_lexicon_file": "drug_lexicon.txt",  # si existe en profiles/, se cargará
        "drug_lexicon_extra": [],

        "ner_labels": ["PER", "ORG", "LOC", "GPE", "NORP", "FAC", "MISC"]
    }

    Path("profiles").mkdir(exist_ok=True, parents=True)
    outp = Path("profiles") / f"{profile_name}.yaml"
    with open(outp, "w", encoding="utf-8") as f:
        yaml.safe_dump(profile, f, sort_keys=False, allow_unicode=True)
    print(f"\nPerfil guardado: {outp}")
    return str(outp)


# ============================
# Ejecución
# ============================
def run_excel_with_profile(xlsx_path: str, profile_path: str, out_path: str = None,
                           dry_run: bool = False, preview: int = 0):
    with open(profile_path, "r", encoding="utf-8") as f:
        prof_raw = yaml.safe_load(f)

    P = normalize_profile(prof_raw, profile_path)

    df = pd.read_excel(xlsx_path)

    nlp_main = load_main_nlp(P["model_path"])
    nlp_sec = load_secondary_nlp()

    # Turbo preview: desactiva NER secundario (es_core_news_md/lg) cuando es preview
    if preview and preview > 0:
        nlp_sec = None

    tag_map = P["tag_map"]
    regex_cfg = P["regex_map"]
    strat = P["masking_strategy"]
    ner_labels = P["ner_labels"]
    gazetteer = P["gazetteer"]
    drug_lexicon = P["drug_lexicon"]

    audit = {"per_col": {}, "per_ent": {}, "rows": int(len(df))}
    def _inc(d, k, n=1): d[k] = int(d.get(k, 0) + n)

    # TEXTO
    for col in P["text_cols"]:
        if col not in df.columns:
            print(f"[AVISO] Columna de texto no encontrada: {col}")
            continue
        new_vals = []
        ner_tot, rx_tot, gaz_tot, heur_tot, prot_tot = {}, {}, 0, 0, 0
        for val in df[col].astype(str).fillna(""):
            anon, counts = anonymize_text(val, nlp_main, nlp_sec, tag_map, regex_cfg,
                                          ner_labels, gazetteer, drug_lexicon, strat)
            new_vals.append(anon)
            for k, v in counts["ner"].items(): _inc(ner_tot, k, v)
            for k, v in counts["regex"].items(): _inc(rx_tot, k, v)
            gaz_tot += counts["gazetteer"]
            heur_tot += counts["heur"]
            prot_tot += counts.get("protected_drug_spans", 0)
            print(f"[INFO] Columna {col}: spans de fármacos protegidos = {prot_tot}")
        df[col] = new_vals
        for k, v in ner_tot.items(): _inc(audit["per_ent"], k, v)
        _inc(audit["per_col"], col, sum(ner_tot.values()) + sum(rx_tot.values()) + gaz_tot + heur_tot)
        _inc(audit, "protected_drug_spans_total", prot_tot)

    # ESTRUCTURADAS
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
        "ner_labels": P["ner_labels"],
        "regex": P["regex_map"],
        "gazetteer_size": len(P["gazetteer"]),
        "masking": {"strategy": P["masking_strategy"], "tag_map": P["tag_map"]},
        "secondary_ner_loaded": bool(nlp_sec is not None),
        "drug_lexicon_size": len(P["drug_lexicon"])
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
    ap = argparse.ArgumentParser(description="ANONIM Excel (mapeo dinámico + auditoría, con protección de fármacos)")
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
