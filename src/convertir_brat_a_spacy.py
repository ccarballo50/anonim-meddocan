# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from collections import Counter

import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans


def parse_ann_file(ann_path):
    """Parsea un .ann BRAT y devuelve lista de (start, end, label)."""
    spans = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Formato típico: T1<TAB>LABEL start end<TAB>texto
            if "\t" in line:
                try:
                    tid, meta, text = line.split("\t", 2)
                    parts = meta.split()
                    if len(parts) >= 3:
                        label = parts[0]
                        start = int(parts[1])
                        end = int(parts[2])
                        spans.append((start, end, label))
                        continue
                except Exception:
                    # Si falla, intentamos el modo "a pelo" de abajo
                    pass

            # Modo más bruto: T1 LABEL start end ...
            parts = line.split()
            if len(parts) >= 4 and parts[0].startswith("T"):
                try:
                    label = parts[1]
                    start = int(parts[2])
                    end = int(parts[3])
                    spans.append((start, end, label))
                except Exception:
                    continue

    return spans


def convert_brat_dirs(nlp, brat_dirs, out_path):
    """Convierte varios directorios BRAT en un único DocBin .spacy."""
    docbin = DocBin(store_user_data=False)
    ents_total = 0
    docs_total = 0
    label_counts = Counter()
    spans_problematic = 0
    spans_discarded_overlap = 0

    for brat_dir in brat_dirs:
        brat_dir = Path(brat_dir)
        if not brat_dir.is_dir():
            print(f"⚠ Directorio no encontrado: {brat_dir}")
            continue

        print(f"📂 Procesando: {brat_dir}")

        for fname in os.listdir(brat_dir):
            if not fname.endswith(".txt"):
                continue

            txt_path = brat_dir / fname
            ann_path = txt_path.with_suffix(".ann")

            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()

            doc = nlp(text)
            spans = []

            if ann_path.exists():
                for start, end, label in parse_ann_file(ann_path):
                    # STRICT: si no alinea exactamente con tokens, se descarta
                    span = doc.char_span(
                        start,
                        end,
                        label=label,
                        alignment_mode="strict",
                    )
                    if span is None:
                        spans_problematic += 1
                        continue
                    spans.append(span)
                    label_counts[label] += 1
                    ents_total += 1

            # Eliminar solapamientos entre spans
            n_before = len(spans)
            spans = filter_spans(spans)
            spans_discarded_overlap += n_before - len(spans)

            doc.ents = spans
            docbin.add(doc)
            docs_total += 1

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    docbin.to_disk(out_path)

    print(f"\n✅ Guardado: {out_path}")
    print(f"   Documentos: {docs_total}")
    print(f"   Entidades totales (antes de filtros): {ents_total}")
    print(f"   Spans problemáticos (no alinean STRICT): {spans_problematic}")
    print(f"   Spans descartados por solapamiento: {spans_discarded_overlap}")
    print(f"   Etiquetas finales: {label_counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True,
                        help="Carpeta raíz que contiene train/dev/test/brat y train_farmacos_brat")
    args = parser.parse_args()

    root = Path(args.data_root)

    print("🧠 Iniciando modelo en blanco (es)...")
    nlp = spacy.blank("es")

    train_dirs = [
        root / "train" / "brat",
        root / "train_farmacos_brat",
    ]
    dev_dirs = [
        root / "dev" / "brat",
    ]
    test_dirs = [
        root / "test" / "brat",
    ]

    print("\n=== TRAIN ===")
    convert_brat_dirs(nlp, train_dirs, root / "train" / "train.spacy")

    print("\n=== DEV ===")
    convert_brat_dirs(nlp, dev_dirs, root / "dev" / "dev.spacy")

    print("\n=== TEST ===")
    convert_brat_dirs(nlp, test_dirs, root / "test" / "test.spacy")

    print("\n🎉 Conversión completa")


if __name__ == "__main__":
    main()
