import argparse
import random
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch, compounding, load_config


def cargar_docbin(path, vocab):
    db = DocBin().from_disk(path)
    docs = list(db.get_docs(vocab))
    return docs


def crear_ejemplos(docs):
    ejemplos = []
    for doc in docs:
        ents = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        ejemplos.append(Example.from_dict(doc, {"entities": ents}))
    return ejemplos


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento seguro ANONIM (evita E024).")
    parser.add_argument("--config", type=str, default="config.cfg", help="Ruta a config.cfg")
    parser.add_argument("--train", type=str, required=True, help="Ruta a train.spacy")
    parser.add_argument("--dev", type=str, required=True, help="Ruta a dev.spacy")
    parser.add_argument("--out", type=str, required=True, help="Carpeta de salida del modelo")
    parser.add_argument("--epochs", type=int, default=30, help="Número de épocas")
    args = parser.parse_args()

    config_path = Path(args.config)
    train_path = Path(args.train)
    dev_path = Path(args.dev)
    out_dir = Path(args.out)

    print(f"📄 Config: {config_path}")
    print(f"📦 Train: {train_path}")
    print(f"📦 Dev  : {dev_path}")
    print(f"📂 Out  : {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n🔧 Cargando configuración y creando pipeline...")
    config = load_config(config_path)
    nlp = spacy.util.load_model_from_config(config, auto_fill=True)

    print("📥 Cargando DocBin de train y dev...")
    train_docs = cargar_docbin(train_path, nlp.vocab)
    dev_docs = cargar_docbin(dev_path, nlp.vocab)

    print(f"   Nº docs train: {len(train_docs)}")
    print(f"   Nº docs dev  : {len(dev_docs)}")

    train_examples = crear_ejemplos(train_docs)
    dev_examples = crear_ejemplos(dev_docs)

    print("\n🚀 Inicializando pesos con ejemplos de entrenamiento...")
    optimizer = nlp.initialize(get_examples=lambda: train_examples)

    n_epochs = args.epochs

    for epoch in range(1, n_epochs + 1):
        random.shuffle(train_examples)
        losses = {}
        skipped_e024 = 0

        batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            for eg in batch:
                try:
                    nlp.update([eg], sgd=optimizer, drop=0.2, losses=losses)
                except ValueError as e:
                    # Aquí evitamos que E024 reviente todo el entrenamiento
                    if "E024" in str(e):
                        skipped_e024 += 1
                        continue
                    else:
                        raise e

        print(f"Época {epoch:02d}/{n_epochs} - pérdidas: {losses} - ejemplos saltados por E024: {skipped_e024}")

        # Evaluación simple en dev al final de cada época
        if epoch == n_epochs:
            print("\n📊 Evaluando modelo en dev (época final)...")
            scores = nlp.evaluate(dev_examples)
            print(f"   Precisión (P): {scores['ents_p']:.3f}")
            print(f"   Recall (R)   : {scores['ents_r']:.3f}")
            print(f"   F1           : {scores['ents_f']:.3f}")

    print(f"\n💾 Guardando modelo entrenado en: {out_dir}")
    nlp.to_disk(out_dir)
    print("✅ Entrenamiento seguro finalizado.")


if __name__ == "__main__":
    main()
