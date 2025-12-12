import os

# Carpetas con anotaciones BRAT
BRAT_DIRS = [
    r"C:\Users\lupem\ANONIM_MEDDOCAN\WEB meddocan\train\brat",
    r"C:\Users\lupem\ANONIM_MEDDOCAN\WEB meddocan\dev\brat",
    r"C:\Users\lupem\ANONIM_MEDDOCAN\WEB meddocan\test\brat",
    r"C:\Users\lupem\ANONIM_MEDDOCAN\WEB meddocan\train_farmacos_brat",
]


def parse_ann(path):
    ents = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("T"):
                continue
            # Formato típico:
            # T1<TAB>LABEL start end [end2 ...]<TAB>text
            try:
                tid, rest = line.split("\t", 1)
            except ValueError:
                print(f"⚠ Línea rara (sin tab) en {path}:\n   {line}")
                continue

            parts = rest.split("\t")[0].split()
            if len(parts) < 3:
                print(f"⚠ Línea rara (pocos campos) en {path}:\n   {line}")
                continue

            label = parts[0]
            # Ojo con spans discontinuos. Solo usamos el primer tramo start/end
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                print(f"⚠ No puedo parsear offsets en {path}:\n   {line}")
                continue

            ents.append((start, end, label, line))
    return ents


def find_overlaps_in_file(path):
    ents = parse_ann(path)
    if len(ents) < 2:
        return

    # Ordenar por inicio, luego fin
    ents_sorted = sorted(ents, key=lambda x: (x[0], x[1]))
    overlaps = []

    for i in range(1, len(ents_sorted)):
        s1, e1, lab1, line1 = ents_sorted[i - 1]
        s2, e2, lab2, line2 = ents_sorted[i]
        # Solape si el inicio del segundo está antes del final del primero
        if s2 < e1:
            overlaps.append((line1, line2))

    if overlaps:
        print(f"\n⚠ SOLAPES en: {path}")
        for l1, l2 in overlaps:
            print("   ", l1)
            print("   ", l2)


def main():
    for brat_dir in BRAT_DIRS:
        print(f"\n🔍 Revisando carpeta BRAT: {brat_dir}")
        if not os.path.isdir(brat_dir):
            print("   (no existe)")
            continue

        for fname in os.listdir(brat_dir):
            if not fname.endswith(".ann"):
                continue
            full = os.path.join(brat_dir, fname)
            find_overlaps_in_file(full)


if __name__ == "__main__":
    main()
