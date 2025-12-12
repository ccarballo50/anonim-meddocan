import os
import glob
from pathlib import Path
from collections import Counter

import spacy


def cargar_spans_gold(ann_path, etiqueta_objetivo="FARMACO"):
    """
    Lee un archivo .ann en formato BRAT y devuelve la lista de spans (start, end)
    para la etiqueta indicada (por defecto FARMACO).
    Ignora líneas mal formateadas.
    """
    spans = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Formato típico: T1\tFARMACO 10 21\tparacetamol
            parts = line.split("\t")
            if len(parts) < 2:
                # Línea rara, la ignoramos
                continue

            tag_info = parts[1]  # "FARMACO 10 21"
            tag_parts = tag_info.split()
            if len(tag_parts) < 3:
                # También raro, la ignoramos
                continue

            label = tag_parts[0]
            if label != etiqueta_objetivo:
                continue

            try:
                start = int(tag_parts[1])
                end = int(tag_parts[2])
            except ValueError:
                # Si los índices no son enteros, ignoramos
                continue

            spans.append((start, end))
    return spans


def evaluar_farmaco_sobre_brat(
    modelo_path,
    brat_dir,
    etiqueta_objetivo="FARMACO",
):
    """
    Evalúa el modelo spaCy sobre un conjunto de archivos BRAT (.txt + .ann)
    calculando precisión, recall y F1 para la etiqueta FARMACO.
    """
    print(f"Cargando modelo desde: {modelo_path}")
    nlp = spacy.load(modelo_path)

    txt_files = sorted(glob.glob(os.path.join(brat_dir, "*.txt")))
    if not txt_files:
        print(f"No se han encontrado .txt en: {brat_dir}")
        return

    tp = 0
    fp = 0
    fn = 0

    por_doc = []

    for txt_path in txt_files:
        base = os.path.splitext(os.path.basename(txt_path))[0]
        ann_path = os.path.join(brat_dir, base + ".ann")
        if not os.path.exists(ann_path):
            # Si no hay anotación, saltamos
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Gold
        gold_spans = cargar_spans_gold(ann_path, etiqueta_objetivo=etiqueta_objetivo)
        gold_spans_set = set(gold_spans)

        # Predicciones
        doc = nlp(text)
        pred_spans = [
            (ent.start_char, ent.end_char)
            for ent in doc.ents
            if ent.label_ == etiqueta_objetivo
        ]
        pred_spans_set = set(pred_spans)

        # True positives: intersección exacta de spans
        tp_doc = len(gold_spans_set & pred_spans_set)
        # False positives: predichos que no están en gold
        fp_doc = len(pred_spans_set - gold_spans_set)
        # False negatives: gold que no se han predicho
        fn_doc = len(gold_spans_set - pred_spans_set)

        tp += tp_doc
        fp += fp_doc
        fn += fn_doc

        por_doc.append(
            {
                "doc": base,
                "gold": len(gold_spans),
                "pred": len(pred_spans),
                "tp": tp_doc,
                "fp": fp_doc,
                "fn": fn_doc,
            }
        )

    # Métricas globales
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    print("\n===== Evaluación FARMACO en subcorpus BRAT =====")
    print(f"Carpeta BRAT: {brat_dir}")
    print(f"N docs evaluados: {len(por_doc)}")
    print(f"Total GOLD spans FARMACO: {tp + fn}")
    print(f"Total spans predichos FARMACO: {tp + fp}")
    print(f"TP: {tp}  FP: {fp}  FN: {fn}")
    print(f"\nPrecisión (P): {precision:.3f}")
    print(f"Sensibilidad (R): {recall:.3f}")
    print(f"F1: {f1:.3f}")


if __name__ == "__main__":
    # Ajusta estas rutas a tu estructura real
    BASE_DIR = r"C:\Users\lupem\ANONIM_MEDDOCAN"
    MODELO_PATH = os.path.join(BASE_DIR, "modelo_meddocan_safe")
    BRAT_FARMACOS_DIR = os.path.join(
        BASE_DIR, r"WEB meddocan", "train_farmacos_brat"
    )

    evaluar_farmaco_sobre_brat(
        modelo_path=MODELO_PATH,
        brat_dir=BRAT_FARMACOS_DIR,
        etiqueta_objetivo="FARMACO",
    )
