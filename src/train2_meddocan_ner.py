

import argparse
import json
from pathlib import Path

from spacy.cli.init_config import init_config
from spacy.cli.train import train as spacy_train
from spacy.cli.evaluate import evaluate as spacy_evaluate


def build_config(cfg_path: Path, train_spacy: Path, dev_spacy: Path) -> None:
    """Genera un config limpio con init_config y ajusta rutas train/dev."""
    print("🧩 Generando config con init_config (tok2vec + ner, eficiencia)...")

    cfg = init_config(
        lang="es",
        pipeline=["tok2vec", "ner"],
        optimize="efficiency",
    )

    # Rutas a los corpora en formato .spacy
    cfg["paths"]["train"] = train_spacy.as_posix()
    cfg["paths"]["dev"] = dev_spacy.as_posix()

    # Aseguramos que el entrenamiento use estos corpora
    if "corpora" not in cfg:
        cfg["corpora"] = {}
    cfg["corpora"]["train"] = {
        "@readers": "spacy.Corpus.v1",
        "path": "${paths.train}",
        "gold_preproc": False,
        "max_length": 0,
        "limit": 0,
    }
    cfg["corpora"]["dev"] = {
        "@readers": "spacy.Corpus.v1",
        "path": "${paths.dev}",
        "gold_preproc": False,
        "max_length": 0,
        "limit": 0,
    }

    # Semilla para reproducibilidad
    cfg.setdefault("system", {})
    cfg["system"]["seed"] = 42

    cfg["training"]["train_corpus"] = "corpora.train"
    cfg["training"]["dev_corpus"] = "corpora.dev"

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.to_disk(cfg_path)
    print(f"✔ Config guardada en: {cfg_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help='Carpeta que contiene "train/train.spacy", "dev/dev.spacy", "test/test.spacy"',
    )
    parser.add_argument(
        "--out",
        type=str,
        default="modelo_meddocan_safe_farmacos",
        help="Carpeta de salida para el modelo entrenado y resultados.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    train_spacy = data_root / "train" / "train.spacy"
    dev_spacy = data_root / "dev" / "dev.spacy"
    test_spacy = data_root / "test" / "test.spacy"

    # Comprobaciones básicas
    for p in [train_spacy, dev_spacy, test_spacy]:
        if not p.is_file():
            raise FileNotFoundError(f"No se encuentra el archivo esperado: {p}")

    cfg_path = out_root / "config_meddocan.cfg"

    # (Re)generar config cada vez, para evitar heredar cosas raras
    if cfg_path.exists():
        print(f"ℹ Eliminando config antiguo: {cfg_path}")
        cfg_path.unlink()
    build_config(cfg_path, train_spacy, dev_spacy)

    # Entrenamiento
    print("\n🚀 Entrenando modelo NER (CPU, sin GPU)...")
    spacy_train(
        config_path=str(cfg_path),
        output_path=str(out_root),
        overrides={},
        use_gpu=-1,   # -1 = CPU
    )

    model_best = out_root / "model-best"
    if not model_best.is_dir():
        raise RuntimeError("❌ No se ha encontrado model-best tras el entrenamiento.")

    # ==========================================
    # 4. EVALUACIÓN FINAL EN TEST Y GUARDADO JSON
    # ==========================================
    print("\n📊 Evaluando en conjunto TEST...")

    scores = spacy_evaluate(nlp, test_data)

    # Convertimos a JSON “bonito” para consola
    print(json.dumps(scores, indent=2, ensure_ascii=False))

    # Guardamos resultados de TEST en la carpeta del modelo
    results_path = out_root / "results_test.json"
    try:
        results_path.write_text(
            json.dumps(scores, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"\n💾 Resultados TEST guardados en: {results_path}")
    except Exception as e:
        print(f"⚠ No se pudo guardar results_test.json: {e}")

    print("\n✅ Proceso completo finalizado.")
    print(f"📁 Modelo entrenado en: {model_best}")

if __name__ == "__main__":
    main()
