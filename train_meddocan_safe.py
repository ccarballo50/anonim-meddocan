import random
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.training import Example


# RUTAS A TUS FICHEROS .SPACY  (AJÚSTALAS SI HACE FALTA)
TRAIN_SPACY = Path(
    r"C:\Users\lupem\ANONIM_MEDDOCAN\WEB meddocan\train\train_no_e024.spacy"
)
DEV_SPACY = Path(
    r"C:\Users\lupem\ANONIM_MEDDOCAN\WEB meddocan\dev\dev.spacy"
)

# CARPETA DE SALIDA PARA EL MODELO ENTRENADO
OUTPUT_DIR = Path(
    r"C:\Users\lupem\ANONIM_MEDDOCAN\modelo_meddocan_safe"
)


def load_docbin(path: Path, nlp):
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el fichero: {path}")
    db = DocBin().from_disk(path)
    docs = list(db.get_docs(nlp.vocab))
    return docs


def main():
    print("📥 Inicializando modelo en blanco: es")
    nlp = spacy.blank("es")

    # Añadimos solo el componente NER
    ner = nlp.add_pipe("ner")

    # Cargamos docs de entrenamiento y dev
    print(f"📂 Cargando train desde: {TRAIN_SPACY}")
    train_docs = load_docbin(TRAIN_SPACY, nlp)
    print(f"   → {len(train_docs)} documentos de entrenamiento")

    print(f"📂 Cargando dev desde: {DEV_SPACY}")
    dev_docs = load_docbin(DEV_SPACY, nlp)
    print(f"   → {len(dev_docs)} documentos de validación")

    # Recogemos todas las etiquetas de las entidades
    labels = set()
    for doc in train_docs:
        for ent in doc.ents:
            labels.add(ent.label_)

    print(f"🏷  Etiquetas NER encontradas en train: {sorted(labels)}")

    for label in labels:
        ner.add_label(label)

    # Construimos Examples para entrenamiento
    train_examples = []
    for doc in train_docs:
        ents = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        train_examples.append(Example.from_dict(doc, {"entities": ents}))

    dev_examples = []
    for doc in dev_docs:
        ents = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        dev_examples.append(Example.from_dict(doc, {"entities": ents}))

    print("⚙ Inicializando parámetros del modelo con ejemplos de entrenamiento...")
    optimizer = nlp.initialize(lambda: train_examples)

    # Bucle de entrenamiento manual, ejemplo a ejemplo (batch=1)
    n_iter = 30  # puedes ajustar este número
    print(f"🚀 Comenzando entrenamiento manual durante {n_iter} épocas...\n")

    for epoch in range(n_iter):
        random.shuffle(train_examples)
        losses = {}

        skipped_e024 = 0

        for eg in train_examples:
            try:
                # update con un solo ejemplo para evitar problemas de E024
                nlp.update([eg], drop=0.2, sgd=optimizer, losses=losses)
            except ValueError as e:
                # Si aun así aparece E024, saltamos ese ejemplo
                if "E024" in str(e):
                    skipped_e024 += 1
                    continue
                else:
                    # Si es otro error, lo relanzamos
                    raise

        print(
            f"Época {epoch+1:02d}/{n_iter} "
            f"- pérdidas: {losses} "
            f"- ejemplos saltados por E024: {skipped_e024}"
        )

    # Evaluación sencilla en dev
    print("\n📊 Evaluando modelo en el conjunto de validación (dev)...")
    scores = nlp.evaluate(dev_examples)
    ents_p = scores.get("ents_p", 0.0)
    ents_r = scores.get("ents_r", 0.0)
    ents_f = scores.get("ents_f", 0.0)
    print(f"   Precisión (P): {ents_p:.2f}")
    print(f"   Recall (R)   : {ents_r:.2f}")
    print(f"   F1           : {ents_f:.2f}")

    # Guardamos el modelo entrenado
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(OUTPUT_DIR)
    print(f"\n💾 Modelo entrenado guardado en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
