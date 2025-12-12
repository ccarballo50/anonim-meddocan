# make_meddocan_excel.py
# Crea un Excel con TODOS los documentos de MEDDOCAN:
#  - 1 fila por documento
#  - columna 'texto' con el contenido completo
#  - columnas "estructuradas" opcionales con el PRIMER valor encontrado por tipo
# Guarda en:  %USERPROFILE%\ANONIM_MEDDOCAN\data\in\meddocan_full.xlsx

import os
import re
from pathlib import Path
import pandas as pd

# === RUTAS ===========================================================
# Ajusta si tu estructura difiere. En tu caso vimos:
# C:\Users\lupem\ANONIM_MEDDOCAN\meddocan\meddocan\(train|dev|test)\[brat?]\*.txt/*.ann
PROJECT_ROOT = Path(os.path.expandvars(r"%USERPROFILE%\ANONIM_MEDDOCAN"))
# <<--- AQUÍ tu raíz del corpus:
MEDDOCAN_ROOT = PROJECT_ROOT / "WEB meddocan"
OUT_XLSX = PROJECT_ROOT / "data" / "in" / "meddocan_full.xlsx"

# Si tus .txt/.ann estuvieran directamente en ...\meddocan\(train|dev|test)\ (sin "brat"),
# el script ya lo detecta automáticamente más abajo.

# === ETIQUETAS "estructuradas" opcionales ============================
# Tomaremos el PRIMER match por tipo para crear columnas de ejemplo.
STRUCT_PRIORITIES = [
    "NOMBRE_SUJETO_ASISTENCIA",
    "ID_SUJETO_ASISTENCIA",
    "NUMERO_TELEFONO",
    "CORREO_ELECTRONICO",
    "FECHAS",
    "HOSPITAL",
    "CALLE",
    "TERRITORIO",
    "PAIS",
]

def read_brat_doc(txt_path: Path):
    """Lee un par .txt/.ann BRAT y devuelve: (texto, lista_ents)"""
    ann_path = txt_path.with_suffix(".ann")
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    ents = []
    if ann_path.exists():
        for line in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            # Formato típico BRAT:
            # T1\tLABEL start end\tSuperficie
            # o multi-span: LABEL 10 18;20 25   (tomamos el PRIMER tramo)
            if not line or not line.startswith("T"):
                continue
            try:
                _tid, rest = line.split("\t", 1)
                tag, surface = rest.split("\t", 1)
                parts = tag.split()
                label = parts[0].strip()
                # Extrae el primer rango "start end"
                span_part = " ".join(parts[1:])
                m = re.match(r"(\d+)\s+(\d+)", span_part)
                if not m:
                    continue
                start, end = int(m.group(1)), int(m.group(2))
                if 0 <= start < end <= len(text):
                    frag = text[start:end]
                else:
                    # rango fuera de texto (raro): ignora
                    continue
                # Descarta spans vacíos o con solo espacios
                if not frag.strip():
                    continue
                ents.append({"label": label, "start": start, "end": end, "text": frag})
            except Exception:
                # Si una línea no cumple, seguimos
                continue
    return text, ents

def first_by_label(ents, label):
    """Devuelve el primer texto anotado para una etiqueta dada."""
    for e in ents:
        if e["label"] == label:
            return e["text"]
    return ""

def build_rows(split_dir: Path, split_name: str):
    rows = []
    # Busca .txt dentro de la carpeta; si está vacía, no añade filas
    for txt_path in sorted(split_dir.glob("*.txt")):
        doc_id = f"{split_name}/{txt_path.stem}"
        text, ents = read_brat_doc(txt_path)
        row = {
            "doc_id": doc_id,
            "texto": text,  # columna de TEXTO LIBRE completa
            "labels_presentes": ",".join(sorted(set(e["label"] for e in ents)))
        }
        # columnas "estructuradas" (primer match por tipo)
        for lab in STRUCT_PRIORITIES:
            row[lab] = first_by_label(ents, lab)
        rows.append(row)
    return rows

def main():
    if not MEDDOCAN_ROOT.exists():
        raise SystemExit(f"No encuentro MEDDOCAN en: {MEDDOCAN_ROOT}")

    splits = ["train", "dev", "test"]
    all_rows = []

    for sp in splits:
        base = MEDDOCAN_ROOT / sp
        # Muchos paquetes de MEDDOCAN incluyen subcarpeta "brat"; si no existe, usamos la base
        split_dir = base / "brat" if (base / "brat").exists() else base
        rows = build_rows(split_dir, sp)
        print(f"[{sp}] documentos encontrados: {len(rows)}  (en {split_dir})")
        all_rows += rows

    if not all_rows:
        raise SystemExit("No se han encontrado .txt en ninguna partición (train/dev/test). Revisa la ruta.")

    df = pd.DataFrame(all_rows)
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUT_XLSX, index=False)
    print(f"\nExcel creado: {OUT_XLSX}")
    print(f"Filas totales: {len(df)}")
    print("Columnas creadas:", list(df.columns))

if __name__ == "__main__":
    main()

