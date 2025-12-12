import os
import random
from pathlib import Path

# Número de documentos sintéticos a generar
N_DOCS = 10000

# Ruta base del proyecto (ajusta si tu carpeta cambia)
BASE_DIR = Path(r"C:\Users\lupem\ANONIM_MEDDOCAN")
OUTPUT_DIR = BASE_DIR / r"WEB meddocan" / "train_farmacos_brat_v2"

# Lista de fármacos (puedes añadir/quitar)
FARMACOS = [
    "paracetamol", "ibuprofeno", "omeprazol", "enalapril", "amoxicilina",
    "ácido clavulánico", "simvastatina", "atorvastatina", "furosemida",
    "heparina", "Sintrom", "Sintrom®", "clopidogrel", "Nolotil", "Nolotil®",
    "Ventolín", "Ventolin", "ceftriaxona", "metamizol", "Lexatin",
    "diazepam", "lorazepam", "alprazolam", "digoxina", "levotiroxina",
    "sertralina", "tramadol", "Augmentine", "Betadine", "hidroxicloroquina",
    "Adiro", "Adiro®", "insulina", "Urbason", "Amlodipino", "Losartán",
    "pantoprazol"
]

DIAGNOSTICOS = [
    "hipertensión arterial", "fibrilación auricular", "diabetes mellitus",
    "insuficiencia cardiaca", "EPOC", "asma", "síndrome coronario crónico",
    "lumbalgia crónica", "depresión mayor", "trastorno de ansiedad"
]

MOTIVOS = [
    "dolor torácico", "disnea", "fiebre", "mareo", "cefalea",
    "dolor abdominal", "malestar general", "síncope"
]

DOSIS = [
    "500 mg", "1 g", "20 mg", "40 mg", "75 mg", "10 mg",
    "0,5 mg", "100 mg"
]

FRECUENCIAS = [
    "cada 8 horas", "cada 12 horas", "cada 24 horas",
    "por la noche", "por la mañana", "cada 6 horas"
]

VIAS = [
    "v.o.", "v.o. cada 8 horas", "vía intravenosa", "vía oral",
    "vía subcutánea", "vía intramuscular"
]

# Plantillas para 1, 2 o 3 fármacos. Importante: cada placeholder aparece una sola vez.
PLANTILLAS_1 = [
    "Paciente en tratamiento crónico con {f1} por {dx}.",
    "Se pauta {f1} {dosis} {via} por {motivo}.",
    "Toma habitual domiciliaria: {f1}.",
    "En Urgencias se administra {f1} {dosis} {via}.",
    "Se suspende de momento {f1} por posible interacción."
]

PLANTILLAS_2 = [
    "Paciente en tratamiento con {f1} y {f2} por {dx}.",
    "Toma domiciliaria: {f1} por la mañana y {f2} por la noche.",
    "Se pauta {f1} {dosis} y se mantiene {f2} habitual.",
    "Recibe {f1} en urgencias y continúa con {f2} al alta.",
    "Asocia tratamiento crónico con {f1} y {f2} para control de {dx}."
]

PLANTILLAS_3 = [
    "Paciente polimedicado con {f1}, {f2} y {f3} por múltiples comorbilidades.",
    "Toma habitual: {f1}, {f2} y {f3}, sin otros fármacos según refiere.",
    "Se revisa tratamiento: {f1}, {f2} y {f3}, ajustando dosis según función renal.",
    "En domicilio mantiene {f1}, {f2} y {f3}; se valora desprescripción.",
    "En la anamnesis farmacológica constan {f1}, {f2} y {f3} a dosis habituales."
]


def build_text_and_entities():
    """Construye un texto clínico sintético y la lista de entidades FARMACO."""
    tipo = random.choices(
        population=["uno", "dos", "tres"],
        weights=[0.5, 0.3, 0.2],  # 50% 1 fármaco, 30% 2, 20% 3
        k=1
    )[0]

    dx = random.choice(DIAGNOSTICOS)
    motivo = random.choice(MOTIVOS)
    dosis = random.choice(DOSIS)
    via = random.choice(VIAS)

    if tipo == "uno":
        f1 = random.choice(FARMACOS)
        plantilla = random.choice(PLANTILLAS_1)
        texto = plantilla.format(f1=f1, dx=dx, motivo=motivo, dosis=dosis, via=via)
        entidades = [(f1, "FARMACO")]

    elif tipo == "dos":
        f1, f2 = random.sample(FARMACOS, 2)
        plantilla = random.choice(PLANTILLAS_2)
        texto = plantilla.format(f1=f1, f2=f2, dx=dx, motivo=motivo, dosis=dosis, via=via)
        entidades = [(f1, "FARMACO"), (f2, "FARMACO")]

    else:  # "tres"
        f1, f2, f3 = random.sample(FARMACOS, 3)
        plantilla = random.choice(PLANTILLAS_3)
        texto = plantilla.format(f1=f1, f2=f2, f3=f3, dx=dx, motivo=motivo, dosis=dosis, via=via)
        entidades = [(f1, "FARMACO"), (f2, "FARMACO"), (f3, "FARMACO")]

    return texto, entidades


def main():
    # Crear carpeta de salida
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📂 Generando nuevo corpus FARMACO en: {OUTPUT_DIR}")

    for i in range(1, N_DOCS + 1):
        texto, entidades = build_text_and_entities()

        # Rutas de los ficheros
        txt_path = OUTPUT_DIR / f"ej{i}.txt"
        ann_path = OUTPUT_DIR / f"ej{i}.ann"

        # Guardar texto
        with txt_path.open("w", encoding="utf-8") as f_txt:
            f_txt.write(texto)

        # Guardar anotaciones BRAT
        with ann_path.open("w", encoding="utf-8") as f_ann:
            tid = 1
            for ent_text, label in entidades:
                # Buscar la posición del fármaco en el texto
                start = texto.index(ent_text)
                end = start + len(ent_text)
                # Línea BRAT: T1\tFARMACO start end\ttexto
                f_ann.write(f"T{tid}\t{label} {start} {end}\t{ent_text}\n")
                tid += 1

        if i % 1000 == 0:
            print(f"   > Generados {i} documentos...")

    print("✅ Corpus FARMACO v2 generado correctamente.")
    print(f"   Total documentos: {N_DOCS}")
    print(f"   Carpeta: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
