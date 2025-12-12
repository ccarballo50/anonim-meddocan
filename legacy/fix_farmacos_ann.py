import os
import re

# Carpeta con las anotaciones BRAT de fármacos
BRAT_DIR = r"C:\Users\lupem\ANONIM_MEDDOCAN\WEB meddocan\train_farmacos_brat"

# Patrón para extraer múltiple entidades de una sola línea sin tabs, ej:
# "T1 FARMACO 7 18 Amlodipino T2 FARMACO 21 35 Atorvastatina"
PATTERN = re.compile(
    r"(T\d+)\s+FARMACO\s+(\d+)\s+(\d+)\s+(.+?)(?=(?:\s+T\d+\s+FARMACO)|$)",
    re.UNICODE,
)


def fix_ann_file(path: str):
    with open(path, encoding="utf8") as f:
        original = f.read()

    lines = original.splitlines()
    new_lines = []
    changed = False

    for line in lines:
        # Solo tocamos líneas que empiezan por T y NO tienen tabulador
        if line.startswith("T") and "\t" not in line:
            matches = list(PATTERN.finditer(line))
            if not matches:
                # Si no encaja el patrón, dejamos la línea como está y avisamos
                print(f"⚠ No se pudo parsear la línea en {path}:\n   {line}")
                new_lines.append(line)
                continue

            changed = True
            for m in matches:
                tid, start, end, text = m.groups()
                text = text.strip()
                new_line = f"{tid}\tFARMACO {start} {end}\t{text}"
                new_lines.append(new_line)
        else:
            new_lines.append(line)

    if changed:
        # Hacemos backup por si acaso
        backup_path = path + ".bak"
        if not os.path.exists(backup_path):
            os.replace(path, backup_path)
        else:
            # Si ya existe un .bak, lo sobreescribimos igualmente
            os.remove(path)
        with open(path, "w", encoding="utf8") as f:
            f.write("\n".join(new_lines))
        print(f"✅ Arreglado: {path}")
    else:
        print(f"Sin cambios: {path}")


def main():
    if not os.path.isdir(BRAT_DIR):
        print(f"❌ Carpeta no encontrada: {BRAT_DIR}")
        return

    for fname in os.listdir(BRAT_DIR):
        if not fname.endswith(".ann"):
            continue
        full_path = os.path.join(BRAT_DIR, fname)
        fix_ann_file(full_path)


if __name__ == "__main__":
    main()
