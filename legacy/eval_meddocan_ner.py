import json
from pathlib import Path
import spacy
from spacy.scorer import Scorer
from spacy.training import Example

def main():
    base_dir = Path(r"C:\Users\lupem\ANONIM_MEDDOCAN")
    model_dir = base_dir / "modelo_meddocan_safe"
    dev_path = base_dir / r"WEB meddocan\dev\dev.spacy"

    print(f"Cargando modelo desde: {model_dir}")
    nlp = spacy.load(model_dir)

    print(f"Cargando datos de validación: {dev_path}")
    from spacy.tokens import DocBin
    doc_bin = DocBin().from_disk(dev_path)
    gold_docs = list(doc_bin.get_docs(nlp.vocab))

    examples = []
    for gold in gold_docs:
        pred = nlp(gold.text)
        examples.append(Example(pred, gold))

    scorer = Scorer()
    scores = scorer.score(examples)

    results = {
        "overall": {
            "precision": scores["ents_p"],
            "recall": scores["ents_r"],
            "f1": scores["ents_f"],
        },
        "per_label": scores["ents_per_type"],
    }

    out_path = model_dir / "results_dev.json"
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n📊 Resultados generales:")
    print(f"   Precisión (P): {results['overall']['precision']:.3f}")
    print(f"   Recall (R)   : {results['overall']['recall']:.3f}")
    print(f"   F1           : {results['overall']['f1']:.3f}")
    print(f"💾 Resultados por entidad guardados en: {out_path}")

if __name__ == "__main__":
    main()
