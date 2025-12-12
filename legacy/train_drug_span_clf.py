# -*- coding: utf-8 -*-
"""
Entrenamiento NER (token-level, BIO) desde carpetas BRAT (train/dev/test).
Compatible Windows + Python 3.12 + Transformers 4.41+.

Estructura esperada:
  <data_dir>/
      train/*.txt + *.ann
      dev/*.txt + *.ann
      test/*.txt + *.ann

Cada .ann con líneas tipo:
  T1  DRUG 123 130    enalapril
(o con spans múltiples: "DRUG 10 20;30 35")

Uso (ejemplo con tu ruta con espacios):
  python train_drug_span_clf.py ^
    --data_dir "C:\\Users\\lupem\\ANONIM_MEDDOCAN\\WEB meddocan" ^
    --output_dir outputs_ner ^
    --epochs 5 --batch_size 8 --max_length 256

"""

import os
import re
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
from datasets import Dataset, DatasetDict
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

import os, torch
# Desactiva paralelismo del tokenizador (evita sobrecostes en Windows)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Ajusta hilos BLAS/torch (prueba primero con todos los núcleos)
n = max(1, (os.cpu_count() or 4))
torch.set_num_threads(n)
os.environ["OMP_NUM_THREADS"] = str(n)
os.environ["MKL_NUM_THREADS"] = str(n)



# ----------------------
# Lectura BRAT
# ----------------------
def read_text(path_txt: str) -> str:
    with open(path_txt, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_ann(path_ann: str) -> List[Dict[str, Any]]:
    """
    Devuelve lista de entidades con posibles spans múltiples:
    [{"label": "DRUG", "spans": [(start, end), (start2, end2), ...]}]
    Ignora líneas no T*.
    """
    entities = []
    if not os.path.exists(path_ann):
        return entities
    with open(path_ann, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("T"):
                continue
            # Formato típico: T1\tLABEL start end;start2 end2\ttext
            try:
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                # parts[1] = "LABEL start end;start2 end2 ..."
                meta = parts[1]
                meta_parts = meta.split(" ")
                label = meta_parts[0]
                coords = " ".join(meta_parts[1:])
                # coords pueden venir como "start end;start2 end2;..."
                spans = []
                for seg in coords.split(";"):
                    seg = seg.strip()
                    if not seg:
                        continue
                    m = re.match(r"(\d+)\s+(\d+)", seg)
                    if m:
                        s = int(m.group(1))
                        e = int(m.group(2))
                        if e > s:
                            spans.append((s, e))
                if spans:
                    entities.append({"label": label, "spans": sorted(spans)})
            except Exception:
                # Ignora líneas mal formateadas
                continue
    return entities


def load_brat_split(split_dir: str) -> List[Dict[str, Any]]:
    """
    Busca .txt/.ann de forma RECURSIVA (por ejemplo si están en train/brat).
    Devuelve [{"id": base, "text": str, "entities": [...]}, ...]
    """
    examples = []
    if not os.path.isdir(split_dir):
        return examples

    for root, _, files in os.walk(split_dir):
        txt_files = [f for f in files if f.endswith(".txt")]
        for txtf in txt_files:
            base = txtf[:-4]
            path_txt = os.path.join(root, base + ".txt")
            path_ann = os.path.join(root, base + ".ann")
            if not os.path.exists(path_txt):
                continue
            text = read_text(path_txt)
            ents = parse_ann(path_ann) if os.path.exists(path_ann) else []
            examples.append({"id": base, "text": text, "entities": ents})
    return examples



# ----------------------
# Etiquetas BIO
# ----------------------
def collect_entity_types(all_examples: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """Recoge todos los tipos de entidad del conjunto completo."""
    types = set()
    for split, exs in all_examples.items():
        for ex in exs:
            for ent in ex["entities"]:
                types.add(ent["label"])
    return sorted(types)


def build_label_list(entity_types: List[str]) -> List[str]:
    """Construye esquema BIO a partir de los tipos."""
    labels = ["O"]
    for t in entity_types:
        labels.append(f"B-{t}")
        labels.append(f"I-{t}")
    return labels


# ----------------------
# Alineación por offsets
# ----------------------
def encode_and_align(examples: List[Dict[str, Any]],
                     tokenizer,
                     label2id: Dict[str, int],
                     max_length: int,
                     doc_stride: int = 128) -> Dict[str, List]:
    """
    Tokeniza cada texto con offsets y crea labels por token (BIO),
    manejando textos largos con overflow_to_sample_mapping.
    Regresa features para construir un Dataset HF.
    """
    encodings = tokenizer(
        [ex["text"] for ex in examples],
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=doc_stride,
        padding=False,
    )

    all_input_ids = encodings["input_ids"]
    all_attention_mask = encodings["attention_mask"]
    all_offsets = encodings["offset_mapping"]
    sample_map = encodings["overflow_to_sample_mapping"]

    features_input_ids = []
    features_attention_mask = []
    features_labels = []
    features_doc_id = []

    # Preprocesa entidades por documento a lista de spans [(s,e,label), ...]
    doc_spans = []
    for ex in examples:
        spans = []
        for ent in ex["entities"]:
            for (s, e) in ent["spans"]:
                spans.append((s, e, ent["label"]))
        doc_spans.append(spans)

    for i in range(len(all_input_ids)):
        doc_idx = sample_map[i]
        text = examples[doc_idx]["text"]
        spans = doc_spans[doc_idx]  # lista de (s,e,label)
        offsets = all_offsets[i]

        # Inicializa todas las posiciones con -100 (ignorar pérdidas en especiales)
        labels_ids = [-100] * len(offsets)

        # Crea una máscara rápida de entidades por rango
        # Estrategia: para cada token, si el offset (s_tok, e_tok) está dentro de (s_ent, e_ent),
        # marca B- si s_tok == s_ent, I- si está dentro pero no al inicio.
        for j, (s_tok, e_tok) in enumerate(offsets):
            if s_tok == e_tok:  # tokens especiales (CLS/SEP) o padding
                labels_ids[j] = -100
                continue
            tag = "O"
            for (s_ent, e_ent, lab) in spans:
                if e_tok <= s_ent or s_tok >= e_ent:
                    continue  # no overlap
                # Hay solape
                if s_tok == s_ent:
                    tag = f"B-{lab}"
                    break
                else:
                    tag = f"I-{lab}"
                    break
            labels_ids[j] = label2id.get(tag, label2id["O"])

        features_input_ids.append(all_input_ids[i])
        features_attention_mask.append(all_attention_mask[i])
        features_labels.append(labels_ids)
        features_doc_id.append(examples[doc_idx]["id"])

    return {
        "input_ids": features_input_ids,
        "attention_mask": features_attention_mask,
        "labels": features_labels,
        "doc_id": features_doc_id,
    }


# ----------------------
# Métricas (seqeval)
# ----------------------
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    true  = p.label_ids

    out_pred, out_true = [], []
    for pr, tr in zip(preds, true):
        mask = tr != -100
        pr = pr[mask]
        tr = tr[mask]
        out_pred.append([ID2LABEL[i] for i in pr])
        out_true.append([ID2LABEL[i] for i in tr])

    return {
        "precision": precision_score(out_true, out_pred),
        "recall":    recall_score(out_true, out_pred),
        "f1":        f1_score(out_true, out_pred),
    }




# ----------------------
# Main
# ----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="NER (BIO) desde BRAT usando RoBERTa clínico ES")
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Carpeta que contiene train/dev/test con .txt y .ann")
    ap.add_argument("--output_dir", type=str, default="outputs_ner")
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--doc_stride", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    ap.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--report_to", type=str, default="none")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    train_dir = os.path.join(args.data_dir, "train")
    dev_dir = os.path.join(args.data_dir, "dev")
    test_dir = os.path.join(args.data_dir, "test")

    # 1) Cargar ejemplos BRAT
    train_ex = load_brat_split(train_dir)
    dev_ex = load_brat_split(dev_dir)
    test_ex = load_brat_split(test_dir)

    if not train_ex:
        raise RuntimeError(f"No se encontraron .txt/.ann en {train_dir}")

    all_examples = {"train": train_ex, "dev": dev_ex, "test": test_ex}

    # 2) Construir etiquetas BIO a partir de todos los splits
    entity_types = collect_entity_types(all_examples)
    labels = build_label_list(entity_types)
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, ensure_ascii=False, indent=2)

    # 3) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True,
        cache_dir=CACHE_DIR,
        local_files_only=False
    )

    # 4) Convertir ejemplos a features token-level (maneja overflow)
    train_feats = encode_and_align(train_ex, tokenizer, label2id, args.max_length, args.doc_stride)
    dev_feats = encode_and_align(dev_ex, tokenizer, label2id, args.max_length, args.doc_stride) if dev_ex else None
    test_feats = encode_and_align(test_ex, tokenizer, label2id, args.max_length, args.doc_stride) if test_ex else None

    # 5) Crear DatasetDict de HF
    ds_dict = {}
    ds_dict["train"] = Dataset.from_dict(train_feats)
    if dev_feats and len(dev_feats["input_ids"]) > 0:
        ds_dict["validation"] = Dataset.from_dict(dev_feats)
    if test_feats and len(test_feats["input_ids"]) > 0:
        ds_dict["test"] = Dataset.from_dict(test_feats)
    ds = DatasetDict(ds_dict)

    # 6) Modelo
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        cache_dir=CACHE_DIR,
    )

    # 7) Collator y métricas
    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics = compute_metrics_builder(id2label)

    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        # ⚡ Entrenamiento
        num_train_epochs=args.epochs,
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        # ⚖️ Evaluación
        eval_strategy="epoch",      # reemplaza evaluation_strategy (deprecado)
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",

        # 🧠 Optimización en CPU
        dataloader_num_workers=2,       # 2 es estable en Windows
        dataloader_pin_memory=False,    # evita sobrecarga en CPU
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,    # reduce memoria de modelo

        # 📏 Secuencia
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,

        # 🌍 Reproducibilidad
        seed=42
    )


    # 9) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"] if "validation" in ds else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if "validation" in ds else None,
    )

    # 10) Train
    trainer.train()

    # 11) Eval (si hay dev)
    if "validation" in ds:
        eval_metrics = trainer.evaluate()
        print("DEV metrics:", eval_metrics)

    # 12) Test (si hay test)
    if "test" in ds:
        preds = trainer.predict(ds["test"])
        print("TEST metrics:", preds.metrics)

        # Reporte detallado seqeval (microview)
        # reconstruimos etiquetas legibles ignorando -100
        y_true_all, y_pred_all = [], []
        logits = preds.predictions
        pred_ids = np.argmax(logits, axis=-1)
        for y_true, y_pred in zip(preds.label_ids, pred_ids):
            tl, tp = [], []
            for t, p_ in zip(y_true, y_pred):
                if t == -100:
                    continue
                tl.append(id2label[int(t)])
                tp.append(id2label[int(p_)])
            y_true_all.append(tl)
            y_pred_all.append(tp)
        print(classification_report(y_true_all, y_pred_all, digits=4))

    # 13) Guardar
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Entrenamiento finalizado. Modelo y labels guardados en:", args.output_dir)


if __name__ == "__main__":
    main()
