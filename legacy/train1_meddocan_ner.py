# -*- coding: utf-8 -*-
"""
train2_meddocan_ner.py
Convierte carpetas BRAT (train/dev/test) -> .spacy, entrena un NER spaCy y evalúa en test.
Uso:
  python train2_meddocan_ner.py --data-root "WEB meddocan" --out "out_real_model"
Requisitos:
  spacy==3.8.7, srsly==2.4.8, typer==0.9.0, click==8.1.3 (recomendado)
"""

import argparse
import json
import sys
from pathlib import Path
import spacy
from spacy.tokens import DocBin
from spacy.training.converters import brat2docs
from spacy.util import ensure_path
from spacy.cli.train import train as spacy_train
from spacy.cli.evaluate import evaluate as spacy_evaluate

# -------- utilidades --------
def convert_brat_dir(brat_dir: Path, out_spacy: Path) -> Path:
    brat_dir = brat_dir.resolve()
    out_spacy = out_spacy.resolve()
    out_spacy.parent.mkdir(parents=True, exist_ok=True)

    nlp = spacy.blank("es")
    txts = sorted(brat_dir.glob("*.txt"))
    db = DocBin(store_user_data=True)

    bad = []
    n_docs = 0
    for txt in txts:
        ann = txt.with_suffix(".ann")
        if not ann.exists():
            bad.append((txt.name, "NO_ANN"))
            continue
        try:
            docs = list(brat2docs(nlp, txt, ner_map=None))
            for d in docs:
                db.add(d)
            n_docs += len(docs)
        except Exception as e:
            bad.append((txt.name, f"ERROR:{e}"))

    db.to_disk(out_spacy)
    print(f"✔ {brat_dir.name}: guardado {out_spacy} con {n_docs} docs")

    if bad:
        log = out_spacy.with_suffix(".bad.log")
        with log.open("w", encoding="utf-8") as f:
            for name, why in bad:
                f.write(f"{name}\t{why}\n")
        print(f"⚠ {len(bad)} ficheros ignorados. Revisa: {log}")
    return out_spacy

def write_minimal_config(cfg_path: Path, train_path: Path, dev_path: Path):
    cfg = f"""
[paths]
train = "{train_path.as_posix()}"
dev = "{dev_path.as_posix()}"

[nlp]
lang = "es"
pipeline = ["tok2vec","ner"]
batch_size = 128

[components]

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v2"
width = 96
embed_size = 96
window_size = 1
maxout_pieces = 2
subword_features = true

[components.ner]
factory = "ner"

[training]
optimizer = {{"@optimizers":"Adam.v1","learn_rate":0.001}}
dropout = 0.2
patience = 1600
max_steps = 20000
eval_frequency = 200

[training.batcher]
@batchers = "spacy.batch_by_padded.v1"
discard_oversize = true
size = 2000
buffer = 256

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"

[initialize]
vectors = null
"""
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(cfg.strip(), encoding="utf-8")
    print(f"✔ Config generada: {cfg_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True,
                    help="Carpeta que contiene train/dev/test (cada una con subcarpeta 'brat').")
    ap.add_argument("--out", type=str, default="out_real_model",
                    help="Carpeta de salida para el modelo y resultados.")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Convertir BRAT -> .spacy
    train_brat = data_root / "train" / "brat"
    dev_brat   = data_root / "dev" / "brat"
    test_brat  = data_root / "test" / "brat"

    if not train_brat.is_dir() or not dev_brat.is_dir() or not test_brat.is_dir():
        print("❌ Estructura esperada: <data-root>/(train|dev|test)/brat/*.txt+*.ann", file=sys.stderr)
        sys.exit(1)

    train_spacy = data_root / "train" / "train.spacy"
    dev_spacy   = data_root / "dev" / "dev.spacy"
    test_spacy  = data_root / "test" / "test.spacy"

    print("==> Convirtiendo BRAT a .spacy …")
    convert_brat_dir(train_brat, train_spacy)
    convert_brat_dir(dev_brat,   dev_spacy)
    convert_brat_dir(test_brat,  test_spacy)

    # 2) Config mínima y entrenamiento
    cfg_path = out_root / "config_autogen.cfg"
    write_minimal_config(cfg_path, train_spacy, dev_spacy)

    print("==> Entrenando spaCy …")
    # spacy_train espera rutas tipo str
    spacy_train(
        config_path=str(cfg_path),
        output_path=str(out_root),
        overrides={},  # ya pusimos paths en la config
        use_gpu=-1,    # -1 = CPU, 0 = primera GPU si existe
        stdout=sys.stdout,
    )
    model_best = out_root / "model-best"
    if not model_best.is_dir():
        print("❌ No se encontró model-best tras el entrenamiento.", file=sys.stderr)
        sys.exit(2)

    # 3) Evaluación en test
    results_path = out_root / "results.json"
    print("==> Evaluando en TEST …")
    scores = spacy_evaluate(
        model=str(model_best),
        data_path=str(test_spacy),
        output=str(results_path),
        use_gpu=-1,
        silent=False,
        displacy_path=None,
        gold_preproc=False,
        ignore_warnings=False,
    )
    # Por si evaluate no escribe, guardamos nosotros también:
    try:
        results_path.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    print("\n✅ Proceso finalizado.")
    print(f"   Modelo: {model_best}")
    print(f"   Métricas TEST: {results_path}")

if __name__ == "__main__":
    main()
