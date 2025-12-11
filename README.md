# ANONIM v2.0 – Herramienta de Inteligencia Artificial para Anonimización Automática de Texto Clínico en Español

ANONIM v2.0 es una herramienta de anonimización automática de texto clínico en español, diseñada para investigación biomédica y especialmente para servicios de urgencias. El sistema utiliza un modelo spaCy 3.8 de NER (Reconocimiento de Entidades) entrenado sobre:

- El corpus oficial MEDDOCAN (train/dev/test)
- Un subcorpus de fármacos anotado en BRAT
- Un pipeline robusto y reproducible: tok2vec + ner

El proyecto permite:
- Entrenar ANONIM desde cero con spaCy
- Evaluar el rendimiento sobre MEDDOCAN y el subcorpus FARMACO
- Anonimizar texto clínico de manera automática
- Permitir que médicos no entrenados suban un Excel a un Google Colab y descarguen un Excel anonimizado

---

## Estructura del repositorio

ANONIM_MEDDOCAN/
│
├── README.md                     ← Documento principal del proyecto
├── LICENSE                       ← Licencia MIT del proyecto
├── requirements.txt              ← Dependencias necesarias
├── config.cfg                    ← Configuración del modelo spaCy
│
├── src/
│   ├── convertir_brat_a_spacy.py
│   ├── convertir_brat_farmacos_solo.py
│   ├── train_meddocan_ner.py
│   ├── evaluar_meddocan_ner.py
│   ├── anonymize_inference.py
│
├── notebooks/
│   ├── ANONIM_Entrenamiento.ipynb        ← Colab técnico (entrenamiento y evaluación)
│   └── ANONIM_Clinico.ipynb              ← Colab simple para médicos
│
├── data/                                 ← NO INCLUIR datasets reales en el repositorio
│   └── ejemplos/                         ← Ejemplos sintéticos sin datos personales
│
├── models/
│   └── .gitignore                        ← Evita subir modelos pesados
│
└── docs/
    ├── articulo_ANONIM.docx
    ├── tabla_1_ANONIM.docx
    └── arquitectura_ANONIM.md

---

## Instalación

Clonar el repositorio (insertar la URL del repositorio real una vez creado):

git clone https://github.com/USUARIO/REPOSITORIO_ANONIM
cd ANONIM_MEDDOCAN
pip install -r requirements.txt

El proyecto requiere Python 3.10+ para compatibilidad con spaCy 3.8.

---

## Entrenamiento del modelo ANONIM v2.0

1) Convertir anotaciones BRAT a spaCy:

python src/convertir_brat_a_spacy.py --data-root "data/meddocan"

2) Entrenar el modelo:

python -m spacy train config.cfg \
  --output models/modelo_anonim_v2 \
  --paths.train "data/meddocan/train.spacy" \
  --paths.dev   "data/meddocan/dev.spacy"

El entrenamiento generará:

models/modelo_anonim_v2/model-best/
models/modelo_anonim_v2/model-last/

---

## Evaluación del modelo

### Evaluación sobre MEDDOCAN

python -m spacy evaluate \
  models/modelo_anonim_v2/model-best \
  data/meddocan/test.spacy \
  --output models/modelo_anonim_v2/results_test.json

### Evaluación del subcorpus de fármacos

python -m spacy evaluate \
  models/modelo_anonim_v2/model-best \
  data/farmacos_test.spacy \
  --output models/modelo_anonim_v2/results_farmacos.json

---

## Uso del modelo para anonimizar texto

Ejemplo básico:

import spacy
nlp = spacy.load("models/modelo_anonim_v2/model-best")

texto = "Paciente Juan Pérez vive en Calle Mayor 12, Madrid. Se pauta ibuprofeno."
doc = nlp(texto)

anon = texto
for ent in doc.ents:
    anon = anon.replace(ent.text, f"[{ent.label_}]")

print(anon)

Salida:

Paciente [NOMBRE_SUJETO_ASISTENCIA] vive en [CALLE], [TERRITORIO]. Se pauta [FARMACO].

---

## Uso clínico: anonimización de un Excel en Google Colab

Este repositorio incluye dos cuadernos Colab:

- notebooks/ANONIM_Entrenamiento.ipynb (uso técnico)
- notebooks/ANONIM_Clinico.ipynb (uso clínico)

El cuaderno clínico permite:

1. Subir un Excel desde el ordenador
2. Seleccionar la columna con texto clínico
3. Aplicar ANONIM sobre cada fila
4. Descargar un Excel con la columna anonimizada

Enlace al Colab (añadir una vez disponible):

PEGAR_AQUI_ENLACE_COLAB_ANONIM_CLINICO

---

## Enlace al TFM

Añadir aquí el enlace al PDF final del Trabajo de Fin de Máster:

PEGAR_AQUI_ENLACE_PDF_TFM

---

## Licencia

Este proyecto se distribuye bajo la licencia MIT.  
El texto completo de la licencia se encuentra en el archivo `LICENSE` incluido en este repositorio.

---

## Autor

César Carballo  
Trabajo Fin de Máster – Inteligencia Artificial  
(Especificar universidad e institución una vez confirmadas)

