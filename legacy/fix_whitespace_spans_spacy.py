import spacy
from spacy.tokens import DocBin
from pathlib import Path
import shutil

DATA_ROOT = Path(r"C:\Users\lupem\ANONIM_MEDDOCAN\WEB meddocan")

SPACY_FILES = [
    ("train", DATA_ROOT / "train" / "train.spacy"),
    ("dev",   DATA_ROOT / "dev" / "dev.spacy"),
    ("test",  DATA_ROOT / "test" / "test.spacy"),
]

nlp = spacy.blank("es")

def clean_doc(doc):
    """
    Recorta espacios en blanco al inicio/final de cada entidad.
    Si al recortar el span se queda vacío o inválido, se descarta esa entidad.
    """
    new_ents = []
    changed = False

    for ent in doc.ents:
        start = ent.start_char
        end = ent.end_char
        text = doc.text[start:end]

        stripped = text.strip()
        if stripped != text:
            # Hay espacios que sobran
            changed = True
            offset_left = len(text) - len(text.lstrip())
            offset_right = len(text) - len(text.rstrip())
            new_start = start + offset_left
            new_end = end - offset_right

            if new_start >= new_end:
                # Span vacío o roto: se descarta
                continue

            span = doc.char_span(new_start, new_end, label=ent.label_)
            if span is None:
                # No se puede reconstruir bien: descartamos esta entidad
                continue
            new_ents.append(span)
        else:
            new_ents.append(ent)

    if changed:
        doc.ents = new_ents
    return doc, changed

def fix_file(split, path):
    print(f"\n🛠 Corrigiendo .spacy de {split}: {path}")
    if not path.exists():
        print("   ⚠ No existe, lo salto")
        return

    # Backup de seguridad
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy(path, backup)
        print(f"   💾 Backup creado: {backup}")
    else:
        print(f"   ℹ Backup ya existía: {backup}")

    docbin = DocBin().from_disk(path)
    docs = list(docbin.get_docs(nlp.vocab))

    fixed_docbin = DocBin(store_user_data=True)
    total_changed = 0
    total_ents_before = 0
    total_ents_after = 0

    for doc in docs:
        total_ents_before += len(doc.ents)
        doc, changed = clean_doc(doc)
        total_ents_after += len(doc.ents)
        if changed:
            total_changed += 1
        fixed_docbin.add(doc)

    fixed_docbin.to_disk(path)
    print(f"   ✔ Guardado {path}")
    print(f"   👉 Docs modificados: {total_changed}/{len(docs)}")
    print(f"   👉 Entidades antes: {total_ents_before}, después: {total_ents_after}")

if __name__ == "__main__":
    for split, path in SPACY_FILES:
        fix_file(split, path)
