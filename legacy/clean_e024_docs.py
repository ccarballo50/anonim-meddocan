import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from pathlib import Path


# RUTAS DE ENTRADA/SALIDA: AJUSTA SI LAS TIENES DISTINTAS
TRAIN_IN = Path(r"C:\Users\lupem\ANONIM_MEDDOCAN\WEB meddocan\train\train.spacy")
TRAIN_OUT = Path(r"C:\Users\lupem\ANONIM_MEDDOCAN\WEB meddocan\train\train_no_e024.spacy")


def main():
    if not TRAIN_IN.exists():
        print(f"❌ No se encuentra el fichero de entrenamiento: {TRAIN_IN}")
        return

    print(f"📥 Cargando docs desde: {TRAIN_IN}")
    nlp_blank = spacy.blank("es")

    db = DocBin().from_disk(TRAIN_IN)
    docs = list(db.get_docs(nlp_blank.vocab))

    print(f"📊 Total de documentos en train: {len(docs)}")

    good_docs = []
    bad_idxs = []

    for i, doc in enumerate(docs):
        # Extraemos las entidades tal como están en el Doc
        ents = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        # Si no tiene entidades, nunca va a provocar E024: lo dejamos pasar
        if not ents:
            good_docs.append(doc)
            continue

        # Para testear el doc, creamos un modelo en blanco con solo NER
        nlp = spacy.blank("es")
        ner = nlp.add_pipe("ner")

        for _, _, label in ents:
            ner.add_label(label)

        example = Example.from_dict(doc, {"entities": ents})

        try:
            # Inicializamos el modelo solo con este ejemplo y probamos un update
            optimizer = nlp.initialize(lambda: [example])
            nlp.update([example], sgd=optimizer)
            # Si no explota, lo consideramos "bueno"
            good_docs.append(doc)
        except ValueError as e:
            if "E024" in str(e):
                bad_idxs.append(i)
                print("❌ Doc problemático (E024) en índice", i)
                print("   Texto (primeros 200 caracteres):")
                print("   ", repr(doc.text[:200]))
                print("-" * 80)
            else:
                # Cualquier otro error lo re-lanzamos
                raise

    print("✅ Filtrado completado.")
    print(f"   Documentos totales : {len(docs)}")
    print(f"   Documentos buenos  : {len(good_docs)}")
    print(f"   Documentos malos   : {len(bad_idxs)}")

    # Guardamos solo los buenos en un nuevo DocBin
    db_out = DocBin(store_user_data=True)
    for doc in good_docs:
        db_out.add(doc)
    db_out.to_disk(TRAIN_OUT)

    print(f"💾 Guardado train limpio sin E024 en: {TRAIN_OUT}")
    if bad_idxs:
        print("ℹ Si quieres, luego podemos inspeccionar a mano esos docs conflictivos.")
    else:
        print("🎉 No había docs conflictivos, entonces el problema es otra cosa (me avisas).")


if __name__ == "__main__":
    main()
