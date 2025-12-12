# brat2spacy.py
# Conversor BRAT (.txt + .ann) -> spaCy DocBin (.spacy) para MEDDOCAN
# - Soporta subcarpeta "brat" (p.ej. .../train/brat/*.txt) o directamente .../train/*.txt
# - Genera train.spacy, dev.spacy y test.spacy
# - Sanea spans: recorta espacios/puntuación (incl. Unicode), descarta spans inválidos
# - Evita solapes (spaCy NER no admite entidades solapadas)
# - Registra spans inválidos en data\spacy\invalid_spans_report.tsv

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import string
import unicodedata

import spacy
from spacy.tokens import DocBin

# ------------------------- Utilidades de saneado / logging -------------------------

# Registro global de spans inválidos (para depurar)
BAD_SPANS: List[Dict] = []

def char_info(ch: str) -> str:
    """Devuelve info legible de un carácter (código y nombre Unicode)."""
    return f"'{ch}' U+{ord(ch):04X} {unicodedata.name(ch, 'UNKNOWN')}"

# Conjunto extendido de espacios Unicode (NBSP, ZWSP, BOM, etc.)
UNICODE_SPACES = {
    "\u00A0",  # NBSP
    "\u1680", "\u2000", "\u2001", "\u2002", "\u2003", "\u2004", "\u2005",
    "\u2006", "\u2007", "\u2008", "\u2009", "\u200A", "\u202F", "\u205F",
    "\u3000",  # IDEOGRAPHIC SPACE
    "\u200B",  # ZERO WIDTH SPACE
    "\uFEFF",  # BOM
}
WS = set([" ", "\t", "\n", "\r", "\f", "\v"]) | UNICODE_SPACES

def is_space_or_punct(ch: str) -> bool:
    """True si es espacio (incl. unicode) o puntuación (cualquier categoría Unicode 'P*')."""
    if ch in WS or ch.isspace():
        return True
    return unicodedata.category(ch).startswith("P")

def trim_span(text: str, s: int, e: int) -> Tuple[int, int]:
    """Recorta espacios/puntuación en los bordes del span, sin salir de rango."""
    while s < e and is_space_or_punct(text[s]):
        s += 1
    while e > s and is_space_or_punct(text[e - 1]):
        e -= 1
    return s, e

# ------------------------- Entrada BRAT y selección de ficheros -------------------------

def find_txt_files(split_dir: Path) -> List[Path]:
    """Devuelve lista de .txt, aceptando .../split/*.txt o .../split/brat/*.txt."""
    direct = list((split_dir).glob("*.txt"))
    brat = list((split_dir / "brat").glob("*.txt"))
    return brat if brat else direct

def read_brat(txt_path: Path, split_name: str) -> Tuple[str, List[Dict]]:
    """
    Lee un documento BRAT: texto + anotaciones .ann (incluye spans discontinuos 'start end;start end').
    Aplica saneado y descarta spans inválidos registrándolos en BAD_SPANS.
    """
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    ann_path = txt_path.with_suffix(".ann")
    ents: List[Dict] = []

    if ann_path.exists():
        for line in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or not line.startswith("T"):
                continue
            # Formato: T1\tLABEL start end[;start end...]\tMENTION
            try:
                _tid, rest = line.split("\t", 1)
                if "\t" in rest:
                    head, _mention = rest.split("\t", 1)
                else:
                    head, _mention = rest, ""
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

                    # Recorte de bordes (espacios/puntuación, incl. Unicode)
                    start, end = trim_span(text, start, end)
                    if not (0 <= start < end <= len(text)):
                        continue

                    frag = text[start:end]
                    # Normalización ligera solo para chequeo (no tocamos índices)
                    frag_check = (
                        frag.replace("\u00A0", " ")
                            .replace("\u200B", "")
                            .replace("\uFEFF", "")
                    )

                    # Descarta y LOG spans problemáticos
                    def log_bad(reason: str):
                        ctx = text[max(0, start - 30):min(len(text), end + 30)]
                        first = char_info(frag[0]) if frag else ""
                        last = char_info(frag[-1]) if frag else ""
                        BAD_SPANS.append({
                            "split": split_name,
                            "file": str(txt_path),
                            "label": label,
                            "start": start,
                            "end": end,
                            "reason": reason,
                            "first": first,
                            "last": last,
                            "frag": frag.replace("\t", " ").replace("\n", " ").replace("\r", " "),
                            "ctx": ctx.replace("\t", " ").replace("\n", " ").replace("\r", " ")
                        })

                    if not frag_check:
                        log_bad("empty-after-normalize")
                        continue

                    # Borde con espacios visibles
                    if frag_check[0].isspace() or frag_check[-1].isspace():
                        log_bad("isspace-edge")
                        continue

                    # Borde con espacio Unicode o puntuación
                    if is_space_or_punct(frag_check[0]) or is_space_or_punct(frag_check[-1]):
                        log_bad("space/punct-edge")
                        continue

                    # Si strip cambia el texto, hay residuos en bordes (NBSP, etc.)
                    if frag_check != frag_check.strip():
                        log_bad("strip-differs")
                        continue

                    ents.append({"label": label, "start": start, "end": end})

            except Exception:
                # Ignora líneas mal formadas
                continue

    # Orden por inicio/fin
    ents.sort(key=lambda x: (x["start"], x["end"]))
    return text, ents

# ------------------------- Conversión a DocBin -------------------------

def overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """True si a y b se solapan (spaCy NER no permite solapamientos)."""
    return not (a[1] <= b[0] or b[1] <= a[0])

def to_docbin(split_dir: Path, nlp, out_path: Path) -> Dict:
    """
    Convierte todos los .txt/.ann de un split en un DocBin (.spacy).
    - Evita solapes y spans con espacios/puntuación en los bordes.
    - Registra los descartes en BAD_SPANS.
    """
    db = DocBin(store_user_data=False)
    labels_set = set()
    n_docs = 0
    n_ents = 0
    skipped_overlaps = 0
    misaligned = 0

    for txt_path in find_txt_files(split_dir):
        text, ents = read_brat(txt_path, split_dir.name)
        doc = nlp.make_doc(text)

        spans = []
        used: List[Tuple[int, int]] = []  # spans (start_char, end_char) ya ocupados

        for e in ents:
            label = e["label"]
            s, t = e["start"], e["end"]
            labels_set.add(label)

            # Evitar solapes
            if any(u[0] < t and s < u[1] for u in used):
                skipped_overlaps += 1
                BAD_SPANS.append({
                    "split": split_dir.name,
                    "file": str(txt_path),
                    "label": label,
                    "start": s,
                    "end": t,
                    "reason": "overlap",
                    "frag": text[s:t]
                })
                continue

            # Crear span alineado
            span = doc.char_span(s, t, label=label, alignment_mode="contract")
            if span is None:
                span = doc.char_span(s, t, label=label, alignment_mode="expand")
            if span is None:
                misaligned += 1
                continue

            # --- Chequeo final de bordes (evita invalid whitespace entity spans) ---
            stxt = span.text
            if not stxt or stxt != stxt.strip():
                ctx = doc.text[max(0, span.start_char - 30):min(len(doc.text), span.end_char + 30)]
                first = char_info(stxt[0]) if stxt else ""
                last = char_info(stxt[-1]) if stxt else ""
                BAD_SPANS.append({
                    "split": split_dir.name,
                    "file": str(txt_path),
                    "label": label,
                    "start": s,
                    "end": t,
                    "reason": "overlap",
                    "frag": text[s:t],
                    "first": "",  # no disponible aquí
                    "last":  "",  # no disponible aquí
                    "ctx": text[max(0, s-30):min(len(text), t+30)].replace("\t"," ").replace("\n"," ").replace("\r"," ")
                })
                continue
            # -----------------------------------------------------------------------

            spans.append(span)
            used.append((span.start_char, span.end_char))

        # Asignar entidades y añadir al DocBin
        try:
            doc.set_ents(spans)
        except ValueError:
            # Si aún hay conflicto, se eliminan spans solapados restantes
            spans_sorted = sorted(spans, key=lambda x: (x.start_char, x.end_char))
            filtered = []
            last_end = -1
            for sp in spans_sorted:
                if sp.start_char >= last_end:
                    filtered.append(sp)
                    last_end = sp.end_char
                else:
                    skipped_overlaps += 1
            doc.set_ents(filtered)

        db.add(doc)
        n_docs += 1
        n_ents += len(spans)

    db.to_disk(out_path)
    return {
        "docs": n_docs,
        "ents": n_ents,
        "labels": sorted(labels_set),
        "skipped_overlaps": skipped_overlaps,
        "misaligned": misaligned,
        "out": str(out_path)
    }


# ------------------------- Programa principal -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meddocan_root", required=True, help="Ruta con train/dev/test (cada uno con .txt + .ann)")
    ap.add_argument("--out_dir", required=True, help="Carpeta de salida para train.spacy/dev.spacy/test.spacy")
    args = ap.parse_args()

    root = Path(args.meddocan_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizador en blanco de español (rápido, sin dependencias extra)
    nlp = spacy.blank("es")

    report = {}
    for split in ("train", "dev", "test"):
        split_dir = root / split
        if not split_dir.exists():
            report[split] = {"error": f"Split no encontrado: {split_dir}"}
            continue
        out_path = out_dir / f"{split}.spacy"
        report[split] = to_docbin(split_dir, nlp, out_path)

    # Guardar informe general del proceso
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Guardar spans inválidos (si los hubo)
    # Guardar spans inválidos (si los hubo)
    if BAD_SPANS:
        report_path = out_dir / "invalid_spans_report.tsv"
        with report_path.open("w", encoding="utf-8") as f:
            f.write("split\tfile\tlabel\tstart\tend\treason\tfirst\tlast\tfrag\tcontext\n")
            for r in BAD_SPANS:
                frag = r.get("frag", "").replace("\t", " ").replace("\n", " ").replace("\r", " ")
                ctx  = r.get("ctx", "").replace("\t", " ").replace("\n", " ").replace("\r", " ")
                first = r.get("first", "")
                last  = r.get("last", "")
                f.write(
                    f'{r.get("split","")}\t{r.get("file","")}\t{r.get("label","")}\t'
                    f'{r.get("start","")}\t{r.get("end","")}\t{r.get("reason","")}\t'
                    f'{first}\t{last}\t{frag}\t{ctx}\n'
                )
        print(f"\n>> Guardado informe de spans inválidos en: {report_path}")


if __name__ == "__main__":
    main()
