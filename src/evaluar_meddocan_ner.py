import json
from pathlib import Path
import spacy
from spacy.tokens import DocBin
from spacy.scorer import Scorer
from spacy.training import Example

def evaluate_spacy_model(nlp, test_docs):
    scorer = Scorer()
    examples = []

    for doc in test_docs:
        pred = nlp(doc.text)
        # OJO: primero el doc gold, luego la predicción
        examples.append(Example(doc, pred))

    scores = scorer.score(examples)
    return scores


def main():
    # Rutas del modelo y del TEST
    model_path = Path("modelo_meddocan_safe_farmacos/model-last")
    test_path = Path("WEB meddocan/test/test.spacy")

    print(f"Cargando modelo desde: {model_path}")
    nlp = spacy.load(model_path)

    print(f"Cargando TEST desde: {test_path}")
    docbin = DocBin().from_disk(test_path)
    test_docs = list(docbin.get_docs(nlp.vocab))

    print("\n📊 Evaluando en TEST...")
    scores = evaluate_spacy_model(nlp, test_docs)

    # Mostrar en consola
    print(json.dumps(scores, indent=2, ensure_ascii=False))

    # Guardar JSON
    results_path = model_path / "results_test.json"
    results_path.write_text(
        json.dumps(scores, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\n💾 results_test.json guardado en: {results_path}")


if __name__ == "__main__":
    main()
