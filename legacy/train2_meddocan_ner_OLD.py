# -*- coding: utf-8 -*-
"""
train2_meddocan_ner.py
Convierte carpetas BRAT (train/dev/test) -> .spacy, combinando:
- MEDDOCAN: train/brat, dev/brat, test/brat
- FARMACOS extra: train_farmacos_brat (solo en train)
Entrena un NER spaCy y evalúa en test.

Uso:
  python train2_meddocan_ner.py --data-root "WEB meddocan" --out "out_meddocan_full"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import spacy
from spacy.tokens import Doc, DocBin
from spacy.cli.train import train as spacy_train
from spacy.cli.evaluate import evaluate as spacy_evaluate
from spacy.cli.init_config import init_config
from spacy.util import filter_spans

# ---------- utilidades BRAT -> Doc ----------

def parse_ann(ann_path: Path) -> List[Tuple[int, int, str]]:
    """
    Lee un .ann BRAT y devuelve una lista de spans (start, end, label).
    Soporta discontinuos tipo: 'T1\\tLABEL 12 20;25 30\\ttexto...'
    y los divide en varios spans contiguos (spaCy no acepta discontinuos).
    """
    spans = []
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("T"):
                continue
            try:
                _tid, rest = line.split("\t", 1)
                taginfo, _text = rest.split("\t", 1)
                parts = taginfo.split()
                label = parts[0]
                offsets_str = " ".join(parts[1:])  # puede contener ';'
                for chunk in offsets_str.split(";"):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    nums = [x for x in chunk.split() if x.isdigit()]
                    if len(nums) != 2:
                        continue
                    start, end = int(nums[0]), int(nums[1])
                    if end > start:
                        spans.append((start, end, label))
            except Exception:
                continue

    # Ordenar y eliminar solapes de forma simple (dejamos el más largo)
    spans.sort(key=lambda x: (x[0], x[1]))
    pruned = []
    last_end = -1
    for s, e, L in spans:
        if s >= last_end:
            pruned.append((s, e, L))
            last_end = e
        else:
            if pruned and (e - s) > (pruned[-1][1] - pruned[-1][0]):
                pruned[-1] = (s, e, L)
                last_end = e
    return pruned


def brat_to_doc(nlp, text_path: Path) -> Doc:
    txt = text_path.read_text("utf-8", errors="ignore")
    ann_path = text_path.with_suffix(".ann")
    spans = parse_ann(ann_path)

    doc = nlp.make_doc(txt)
    ents = []
    dropped = 0

    for s, e, label in spans:
        # Solo aceptamos spans perfectamente alineados a tokens (sin expand)
        span = doc.char_span(s, e, label=label, alignment_mode="contract")
        if span is None:
            dropped += 1
            continue
        ents.append(span)

    from spacy.util import filter_spans
    doc.ents = filter_spans(ents)

    if dropped:
        # Aviso para depuración (puedes comentarlo si molesta)
        print(f"⚠ {text_path.name}: descartadas {dropped} anotaciones desalineadas")

    return doc



def convert_brat_dirs(dirs: List[Path], out_spacy: Path) -> Path:
    """
    Lee todos los *.txt de una o varias carpetas BRAT, busca su .ann correspondiente
    y genera un DocBin .spacy con todas las entidades.
    """
    out_spacy = out_spacy.resolve()
    out_spacy.parent.mkdir(parents=True, exist_ok=True)

    nlp_tmp = spacy.blank("es")
    db = DocBin(store_user_data=True)

    txt_count = 0
    bad = []

    for d in dirs:
        d = d.resolve()
        if not d.is_dir():
            continue
        txts = sorted(d.glob("*.txt"))
        for txt in txts:
            ann = txt.with_suffix(".ann")
            if not ann.exists():
                bad.append((f"{d.name}/{txt.name}", "NO_ANN"))
                continue
            try:
                doc = brat_to_doc(nlp_tmp, txt)
                db.add(doc)
                txt_count += 1
            except Exception as e:
                bad.append((f"{d.name}/{txt.name}", f"ERROR:{e}"))

    db.to_disk(out_spacy)
    dir_names = ", ".join(d.name for d in dirs if d.is_dir())
    print(f"✔ {dir_names}: guardado {out_spacy} con {txt_count} docs")

    if bad:
        log = out_spacy.with_suffix(".bad.log")
        with log.open("w", encoding="utf-8") as f:
            for name, why in bad:
                f.write(f"{name}\t{why}\n")
        print(f"⚠ {len(bad)} ficheros ignorados. Revisa: {log}")

    return out_spacy

def clean_spacy_file(spacy_path: Path):
    """Limpia un archivo .spacy:
    - elimina entidades con espacios al inicio/fin
    - elimina solapamientos usando filter_spans
    """
    print(f"🧹 Limpiando {spacy_path} ...")
    nlp = spacy.blank("es")
    db_in = DocBin().from_disk(spacy_path)
    db_out = DocBin()

    for doc in db_in.get_docs(nlp.vocab):
        clean_ents = []
        for ent in doc.ents:
            # descartar entidades con espacios al principio o final
            if ent.text != ent.text.strip():
                continue
            clean_ents.append(ent)

        # quitar solapamientos: se queda con las entidades más largas y sin conflicto
        doc.ents = filter_spans(clean_ents)
        db_out.add(doc)

    db_out.to_disk(spacy_path)
    print(f"✔ Limpiado y guardado: {spacy_path}")


def write_config_with_init(cfg_path: Path, train_path: Path, dev_path: Path):
    # 1) Generamos config base con spaCy
    cfg = init_config(
        lang="es",
        pipeline=["ner"],      # pedimos solo NER
        optimize="efficiency"
    )

    # 2) DEBUG: ver qué pipeline ha generado por defecto
    print("Pipeline ORIGINAL de init_config:", cfg["nlp"]["pipeline"])
    print("Componentes ORIGINALES:", list(cfg["components"].keys()))

    # 3) FORZAR pipeline correcto: tok2vec + ner
    cfg["nlp"]["pipeline"] = ["tok2vec", "ner"]

    # 4) Si por alguna razón se ha colado un componente raro 'n', lo eliminamos
    if "n" in cfg["components"]:
        print("⚠ Eliminando componente espurio 'n' del config")
        del cfg["components"]["n"]

    # 5) Aseguramos que exista la entrada 'ner' en components
    if "ner" not in cfg["components"]:
        cfg["components"]["ner"] = {"factory": "ner"}

    # 6) Rutas train/dev
    cfg["paths"]["train"] = train_path.as_posix()
    cfg["paths"]["dev"] = dev_path.as_posix()

    # 7) Semilla y GPU
    cfg["training"]["seed"] = 42
    cfg["training"]["gpu_allocator"] = "pytorch"

    # 8) Guardar config final y mostrar resumen
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.to_disk(cfg_path)
    print(f"✔ Config generada: {cfg_path}")
    print("Pipeline FINAL:", cfg["nlp"]["pipeline"])
    print("Componentes FINALES:", list(cfg["components"].keys()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Carpeta que contiene train/dev/test (cada una con subcarpeta 'brat') "
             "y opcionalmente 'train_farmacos_brat'.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="out_meddocan_full",
        help="Carpeta de salida para el modelo y resultados.",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # MEDDOCAN
    train_brat = data_root / "train" / "brat"
    dev_brat   = data_root / "dev" / "brat"
    test_brat  = data_root / "test" / "brat"

    # FARMACOS extra (solo train)
    farmacos_dir = data_root / "train_farmacos_brat"

    if not train_brat.is_dir() or not dev_brat.is_dir() or not test_brat.is_dir():
        print("❌ Estructura esperada: <data-root>/(train|dev|test)/brat/*.txt+*.ann",
              file=sys.stderr)
        sys.exit(1)

    train_spacy = data_root / "train" / "train_no_e024.spacy"
    dev_spacy   = data_root / "dev" / "dev.spacy"
    test_spacy  = data_root / "test" / "test.spacy"

    # Limpiar los .spacy para evitar anotaciones problemáticas (E024)
    clean_spacy_file(train_spacy)
    clean_spacy_file(dev_spacy)
    clean_spacy_file(test_spacy)


    print("==> Convirtiendo BRAT a .spacy …")

    # TRAIN = MEDDOCAN train/brat + train_farmacos_brat (si existe)
    train_dirs = [train_brat]
    if farmacos_dir.is_dir():
        print(f"✔ Detectada carpeta extra de fármacos: {farmacos_dir}")
        train_dirs.append(farmacos_dir)
    else:
        print("ℹ No se encontró 'train_farmacos_brat'. Solo se usará MEDDOCAN para train.")

    convert_brat_dirs(train_dirs, train_spacy)

    # DEV y TEST = solo MEDDOCAN
    convert_brat_dirs([dev_brat],  dev_spacy)
    convert_brat_dirs([test_brat], test_spacy)

    # 2) Config con init_config y entrenamiento
    cfg_path = out_root / "config_autogen.cfg"
    write_config_with_init(cfg_path, train_spacy, dev_spacy)

    print("==> Entrenando spaCy …")
    spacy_train(
        config_path=str(cfg_path),
        output_path=str(out_root),
        overrides={},
        use_gpu=-1,    # -1 = CPU, 0 = primera GPU si existe
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

    try:
        results_path.write_text(
            json.dumps(scores, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    except Exception:
        pass

    print("\n✅ Proceso finalizado.")
    print(f"   Modelo: {model_best}")
    print(f"   Métricas TEST: {results_path}")


if __name__ == "__main__":
    main()
