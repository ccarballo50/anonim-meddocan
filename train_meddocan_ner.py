import os, re, json, argparse, itertools, random
from typing import List, Dict, Any, Tuple
import numpy as np

from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, TrainingArguments, Trainer)
from seqeval.metrics import precision_score, recall_score, f1_score

random.seed(42); np.random.seed(42)

# --- al inicio del script ---
ENTITIES_MODE = "DRUG_ONLY"

# Se auto-detectan los tipos del corpus y se filtran los de fármaco.
DRUG_CANON = {
    "DRUG","FARMACO","MEDICAMENTO","MED","CHEM","CHEMICAL",
    "ACTIVE_SUBSTANCE","DRUG_OR_ACTIVE"
}


ENTITIES_MODE = "ALL"  # "DRUG_ONLY" para solo fármaco; "ALL" por defecto

def read_text(p):
    with open(p, "r", encoding="utf-8") as f: 
        return f.read()

def parse_ann(path_ann:str):
    ents=[]
    if not os.path.exists(path_ann): return ents
    with open(path_ann,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line or not line.startswith("T"): 
                continue
            try:
                left, _ = line.split("\t",1)
                _tid, span = left.split(" ",1)
                parts = span.split()
                etype = parts[0]
                start = int(parts[1]); end = int(parts[2])
                ents.append((start,end,etype))
            except Exception:
                continue
    return ents

def load_brat_split(split_dir:str):
    out=[]
    if not os.path.isdir(split_dir): return out
    for root, _, files in os.walk(split_dir):
        for fn in files:
            if not fn.endswith(".txt"): continue
            base = fn[:-4]
            ptxt = os.path.join(root, fn)
            pann = os.path.join(root, base + ".ann")
            txt = read_text(ptxt)
            ents = parse_ann(pann)
            if ENTITIES_MODE=="DRUG_ONLY":
                ents = [(s,e,t) for (s,e,t) in ents if t.upper() in {"DRUG","ACTIVE_SUBSTANCE","DRUG_OR_ACTIVE"}]
            out.append({"id": base, "text": txt, "entities": ents})
    return out

def build_label_list(examples):
    types=set()
    for ex in examples:
        for _,_,t in ex["entities"]:
            types.add(t)
    types = sorted(types)
    labels = ["O"] + list(itertools.chain.from_iterable((f"B-{t}", f"I-{t}") for t in types))
    return labels

def char_spans_to_bio(text, entities, tok, max_len):
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=max_len)
    offs = enc["offset_mapping"]
    tags = ["O"]*len(offs)
    for (s,e,t) in entities:
        i = 0
        while i < len(offs):
            a,b = offs[i]
            if b<=s:
                i+=1; continue
            if a>=e:
                break
            if tags[i]=="O":
                tags[i]=f"B-{t}"
                j=i+1
                while j<len(offs):
                    a2,b2=offs[j]
                    if b2<=s or a2>=e: break
                    if tags[j]=="O": tags[j]=f"I-{t}"
                    j+=1
                i=j
            else:
                i+=1
    return enc, tags

def encode_dataset(examples, tok, label2id, max_len):
    inputs = {"input_ids":[], "attention_mask":[], "labels":[]}
    for ex in examples:
        enc, tags = char_spans_to_bio(ex["text"], ex["entities"], tok, max_len)
        ids = [label2id.get(t, 0) for t in tags]  # O->0
        inputs["input_ids"].append(enc["input_ids"])
        inputs["attention_mask"].append(enc["attention_mask"])
        inputs["labels"].append(ids)
    return Dataset.from_dict(inputs)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    true  = p.label_ids
    out_pred, out_true = [], []
    for pr, tr in zip(preds, true):
        mask = tr != -100   # << usar máscara de labels, no attention_mask
        pr = pr[mask]; tr = tr[mask]
        out_pred.append([ID2LABEL[i] for i in pr])
        out_true.append([ID2LABEL[i] for i in tr])
    return {
        "precision": float(precision_score(out_true, out_pred)),
        "recall":    float(recall_score(out_true, out_pred)),
        "f1":        float(f1_score(out_true, out_pred)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default="BSC-LT/roberta-base-biomedical-clinical-es")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--doc_stride", type=int, default=64)
    args = parser.parse_args()

    global ID2LABEL
    train = load_brat_split(os.path.join(args.data_dir,"train"))
    dev   = load_brat_split(os.path.join(args.data_dir,"dev"))

    if not train:
        raise RuntimeError(f"No .txt/.ann en {args.data_dir}/train (¿están en train/brat?)")

    labels = build_label_list(train + dev)
    LABEL2ID = {l:i for i,l in enumerate(labels)}
    ID2LABEL = {i:l for l,i in LABEL2ID.items()}
    print("Etiquetas:", labels)

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    ds_train = encode_dataset(train, tok, LABEL2ID, args.max_length)
    ds_dev   = encode_dataset(dev, tok, LABEL2ID, args.max_length) if dev else None

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model, num_labels=len(labels),
        id2label=ID2LABEL, label2id=LABEL2ID
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        seed=42
    )

    data_collator = DataCollatorForTokenClassification(tok)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if ds_dev:
        metrics = trainer.evaluate()
        print(metrics)
        with open(os.path.join(args.output_dir,"eval_results.json"),"w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

if __name__=="__main__":
    main()
