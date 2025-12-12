# anonim_meddocan_real.py
# Detección "real" (sin mirar el gold en predicción) con regex duras + spaCy NER (modelo entrenado MEDDOCAN)
# Evalúa Precision/Recall/F1 vs BRAT gold y también si el documento queda 100% anonimizado.

import os, re, json, argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
import spacy

# =========================
#    Lectura BRAT (gold)
# =========================
def read_brat_doc(txt_path: Path) -> Tuple[str, List[Dict]]:
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    ann_path = txt_path.with_suffix(".ann")
    ents = []
    if ann_path.exists():
        for line in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or not line.startswith("T"):
                continue
            # Formato: T1\tLABEL start end[;start end...]\tMENTION
            try:
                _tid, rest = line.split("\t", 1)
                head, _mention = rest.split("\t", 1) if "\t" in rest else (rest, "")
                parts = head.split()
                if not parts:
                    continue
                label = parts[0]
                coords = " ".join(parts[1:])
                for seg in coords.split(";"):
                    seg = seg.strip()
                    if not seg:
                        continue
                    s_e = seg.split()
                    if len(s_e) != 2:
                        continue
                    start, end = int(s_e[0]), int(s_e[1])
                    if 0 <= start < end <= len(text):
                        ents.append({"label": label, "start": start, "end": end})
            except Exception:
                # línea ANN malformada: la ignoramos
                pass
    ents.sort(key=lambda x: (x["start"], x["end"]))
    return text, ents


def merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Fusiona intervalos solapados [(s,e), ...] en una lista mínima de intervalos."""
    if not spans:
        return []
    s = sorted(spans, key=lambda x: (x[0], x[1]))
    merged = []
    cs, ce = s[0]
    for a, b in s[1:]:
        if a <= ce:
            ce = max(ce, b)
        else:
            merged.append((cs, ce))
            cs, ce = a, b
    merged.append((cs, ce))
    return merged


# =========================
#       Regex “duras”
# =========================
# (solo identificadores inequívocos; evitamos regex amplias que añaden ruido)
RE_DNI   = re.compile(r"\b(?:\d{7,8}|[XYZ]\d{7,8})[A-Za-z]\b")
RE_NSS   = re.compile(r"\b\d{2}\s?\d{10}\b")
RE_TEL   = re.compile(r"\b(?:\+34\s?)?(?:6|7|8|9)\d{8}\b")
RE_EMAIL = re.compile(r"[A-Za-z0-9_.+\-]+@[A-Za-z0-9\-]+\.[A-Za-z0-9.\-]+")
RE_FECHA = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b")  # fecha simple dd/mm(/aaaa)
RE_URL   = re.compile(r"https?://\S+")
RE_IDHC  = re.compile(r"\b(?:HCE|HC|HIST|EPIS|EXPED|NHC)\s*[:\-]?\s*\w{3,}\b", re.IGNORECASE)

REGEXES = [
    ("ID", RE_DNI),
    ("ID", RE_NSS),
    ("PHONE", RE_TEL),
    ("EMAIL", RE_EMAIL),
    ("DATE", RE_FECHA),
    ("URL", RE_URL),
    ("ID", RE_IDHC),
]

def regex_spans(text: str) -> List[Tuple[int, int, str]]:
    spans = []
    for lab, rx in REGEXES:
        for m in rx.finditer(text):
            spans.append((m.start(), m.end(), lab))
    return spans


# =========================
#  spaCy NER (modelo MEDDOCAN)
# =========================
NLP = None
def load_nlp():
    """Carga perezosa del modelo entrenado (ruta ABSOLUTA)."""
    global NLP
    if NLP is None:
        # ⚠️ Ajusta esta ruta si tu usuario o carpeta cambian:
        NLP = spacy.load(r"C:\Users\lupem\ANONIM_MEDDOCAN\models\ner_meddocan\model-best")
    return NLP

# Recorte conservador de spans largos (direcciones/centros) para no “comerse” conectores
STOP_TOKENS = [",", ";", ".", " y ", " e ", " con ", " sin ", " su ", " sus "]
def clip_right(text: str, start: int, end: int) -> int:
    frag = text[start:end]
    # {coma, punto, punto y coma}
    cuts = [i for i in (frag.find(","), frag.find("."), frag.find(";")) if i != -1]
    # conectores comunes
    for kw in STOP_TOKENS:
        k = frag.find(kw)
        if k != -1:
            cuts.append(k)
    if not cuts:
        return end
    c = min(cuts)
    return start + c if c > 0 else end

def spacy_spans(text: str) -> List[Tuple[int, int, str]]:
    """Devuelve TODAS las entidades que detecta el modelo MEDDOCAN tal cual."""
    nlp = load_nlp()
    doc = nlp(text)
    out = []
    for ent in doc.ents:
        s, e, label = ent.start_char, ent.end_char, ent.label_
        # recorte conservador para spans propensos a “arrastrar” texto no-PII
        if label in ("CALLE", "HOSPITAL", "INSTITUCION", "CENTRO_SALUD"):
            e = clip_right(text, s, e)
        out.append((s, e, label))
    return out


# =========================
#   Fusión de predicciones
# =========================
def unify_pred_spans(text: str) -> List[Tuple[int, int, str]]:
    """
    Combina regex “duras” (prioridad absoluta para PHONE/EMAIL/ID/URL/DATE) con
    las entidades del modelo MEDDOCAN. El resultado se devuelve como spans
    fusionados con etiqueta genérica "PHI" para enmascarado.
    """
    # 1) spans críticos por regex (garantizan anonimización aunque el NER falle)
    hard = regex_spans(text)                       # [(s,e,label), ...]
    hard_intervals = [(s, e) for (s, e, _) in hard]

    # 2) spans del modelo
    ner_spans = spacy_spans(text)                  # [(s,e,label), ...]
    ner_intervals = [(s, e) for (s, e, _) in ner_spans]

    # 3) primero todo lo de regex, luego lo del modelo, y fusionamos
    all_intervals = []
    all_intervals += hard_intervals
    all_intervals += ner_intervals

    merged = merge_spans(all_intervals)
    return [(s, e, "PHI") for (s, e) in merged]


# =========================
#        Enmascarado
# =========================
def mask_predicted(text: str, pred_spans: List[Tuple[int, int, str]]) -> str:
    if not pred_spans:
        return text
    pred_intervals = merge_spans([(s, e) for (s, e, _) in pred_spans])
    out = []
    last = 0
    for (s, e) in pred_intervals:
        out.append(text[last:s])
        out.append("[PHI]")
        last = e
    out.append(text[last:])
    return "".join(out)


# =========================
#         Métricas
# =========================
def strict_match(gold: List[Tuple[int, int]], pred: List[Tuple[int, int]]) -> Tuple[int, int, int]:
    """Exact match (start, end)."""
    gold_set: Set[Tuple[int, int]] = set(gold)
    pred_set: Set[Tuple[int, int]] = set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def overlap_match(gold: List[Tuple[int, int]], pred: List[Tuple[int, int]], iou_thr=0.5) -> Tuple[int, int, int]:
    """Coincidencia por IoU a nivel de carácter (umbral por defecto 0.5)."""
    matched_g = set()
    matched_p = set()
    tp = 0
    for i, (gs, ge) in enumerate(gold):
        gset = set(range(gs, ge))
        best = -1
        best_j = -1
        for j, (ps, pe) in enumerate(pred):
            if j in matched_p:
                continue
            pset = set(range(ps, pe))
            inter = len(gset & pset)
            uni = len(gset | pset)
            iou = inter / uni if uni else 0.0
            if iou > best:
                best = iou
                best_j = j
        if best >= iou_thr:
            tp += 1
            matched_g.add(i)
            matched_p.add(best_j)
    fp = len([j for j in range(len(pred)) if j not in matched_p])
    fn = len([i for i in range(len(gold)) if i not in matched_g])
    return tp, fp, fn


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def eval_doc(text: str, gold_ents: List[Dict]) -> Dict:
    # gold y pred como intervalos consolidados
    gold = merge_spans([(e["start"], e["end"]) for e in gold_ents])
    pred = unify_pred_spans(text)
    pred_intervals = merge_spans([(s, e) for (s, e, _) in pred])

    # Métricas detección (estricto y solapamiento)
    tp_s, fp_s, fn_s = strict_match(gold, pred_intervals)
    p_s, r_s, f1_s = prf(tp_s, fp_s, fn_s)

    tp_o, fp_o, fn_o = overlap_match(gold, pred_intervals, iou_thr=0.5)
    p_o, r_o, f1_o = prf(tp_o, fp_o, fn_o)

    # Anonimización resultante (¿queda algún gold sin ocultar en el texto enmascarado?)
    anon = mask_predicted(text, pred)
    exposed = 0
    for (gs, ge) in gold:
        frag = text[gs:ge]
        if frag and frag in anon:
            exposed += 1
    doc_full = (exposed == 0)

    return {
        "gold_spans": len(gold),
        "pred_spans": len(pred_intervals),
        "strict": {"tp": tp_s, "fp": fp_s, "fn": fn_s, "p": p_s, "r": r_s, "f1": f1_s},
        "overlap": {"tp": tp_o, "fp": fp_o, "fn": fn_o, "p": p_o, "r": r_o, "f1": f1_o},
        "exposed": exposed,
        "doc_fully_anonymized": doc_full,
    }


# =========================
#     Procesamiento split
# =========================
def collect_txt_files(split_dir: Path) -> list:
    direct = list((split_dir).glob("*.txt"))
    brat = list((split_dir / "brat").glob("*.txt"))
    return brat if brat else direct


def process_split(split_dir: Path, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "docs": 0,
        "docs_fully_anonymized": 0,
        "sum_gold": 0,
        "sum_pred": 0,
        "strict_tp": 0,
        "strict_fp": 0,
        "strict_fn": 0,
        "over_tp": 0,
        "over_fp": 0,
        "over_fn": 0,
        "total_exposed": 0,
    }
    details = []
    for txt_path in collect_txt_files(split_dir):
        text, gold = read_brat_doc(txt_path)
        ev = eval_doc(text, gold)

        summary["docs"] += 1
        summary["sum_gold"] += ev["gold_spans"]
        summary["sum_pred"] += ev["pred_spans"]
        summary["strict_tp"] += ev["strict"]["tp"]
        summary["strict_fp"] += ev["strict"]["fp"]
        summary["strict_fn"] += ev["strict"]["fn"]
        summary["over_tp"] += ev["overlap"]["tp"]
        summary["over_fp"] += ev["overlap"]["fp"]
        summary["over_fn"] += ev["overlap"]["fn"]
        summary["total_exposed"] += ev["exposed"]
        if ev["doc_fully_anonymized"]:
            summary["docs_fully_anonymized"] += 1

        details.append({"doc": txt_path.name, **ev})

    # agregados
    def agg(tp, fp, fn):
        p, r, f1 = prf(tp, fp, fn)
        return {"p": p, "r": r, "f1": f1}

    strict_agg = agg(summary["strict_tp"], summary["strict_fp"], summary["strict_fn"])
    over_agg = agg(summary["over_tp"], summary["over_fp"], summary["over_fn"])

    out = {
        "summary": {
            "docs": summary["docs"],
            "docs_fully_anonymized": summary["docs_fully_anonymized"],
            "rate_docs_fully_anonymized": (summary["docs_fully_anonymized"] / summary["docs"])
            if summary["docs"]
            else 0.0,
            "gold_spans": summary["sum_gold"],
            "pred_spans": summary["sum_pred"],
            "strict": strict_agg,
            "overlap": over_agg,
            "total_exposed": summary["total_exposed"],
            "span_exposure_rate": (summary["total_exposed"] / summary["sum_gold"])
            if summary["sum_gold"]
            else 0.0,
        },
        "details": details,
    }
    (out_dir / "eval_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out["summary"]


# =========================
#          Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meddocan_root", required=True, help="Ruta con train/dev/test (.txt + .ann)")
    ap.add_argument("--out_root", required=True, help="Carpeta de salida para reportes")
    args = ap.parse_args()

    med_root = Path(args.meddocan_root)
    out_root = Path(args.out_root)

    for sp in ("train", "dev", "test"):
        split_dir = med_root / sp
        if not split_dir.exists():
            print(f"[AVISO] Falta split {sp}: {split_dir}")
            continue
        res = process_split(split_dir, out_root / sp)
        print(f"== {sp} ==")
        print(json.dumps(res, indent=2, ensure_ascii=False))

    print("== Hecho ==\nRevisa eval_results.json en cada split.")

if __name__ == "__main__":
    main()

