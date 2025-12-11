# -*- coding: utf-8 -*-

import os
from pathlib import Path
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

def parse_ann_file(ann_path):
    spans = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if "\t" in line:
                try:
                    tid, meta, text = line.split("\t", 2)
                    parts = meta.split()
                    label = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])
                    spans.append((start, end, label))
                    continue
                except:
                    pass
            parts = line.split()
            if len(parts) >= 4 and parts[0].startswith("T"):
                try:
                    label = parts[1]
                    start = int(parts[2])
                    end = int(parts[3])
                    spans.append((start, end, label))
                except:
                    continue
    return spans

def main():
    brat_dir = Path("WEB meddocan/train_farmacos_brat")
    out_path = Path("WEB meddocan/farmacos_test.spacy")

    print(f"📂 Convirtiendo BRAT de fármacos en: {brat_dir}")

    nlp = spacy.blank("es")
    docbin = DocBin(store_user_data=False)

    for fname in os.listdir(brat_dir):
        if not fname.endswith(".txt"):
            continue

        txt_path = brat_dir / fname
        ann_path = txt_path.with_suffix(".ann")

        text = txt_path.read_text(encoding="utf-8")
        doc = nlp(text)

        spans = []
        if ann_path.exists():
            for start, end, label in parse_ann_file(ann_path):
                span = doc.char_span(start, end, label=label, alignment_mode="strict")
                if span:
                    spans.append(span)

        spans = filter_spans(spans)
        doc.ents = spans
        docbin.add(doc)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    docbin.to_disk(out_path)

    print(f"✅ Guardado FARMACOS TEST: {out_path}")

if __name__ == "__main__":
    main()
