# ANONIM v2.0  
## Anonimización automática de texto clínico en español mediante PLN (spaCy + NER)

### Trabajo Fin de Máster

---

## 1. Descripción general

**ANONIM v2.0** es una herramienta de anonimización automática de texto clínico en español basada en técnicas de *Procesamiento del Lenguaje Natural (PLN)*, desarrollada como **Trabajo Fin de Máster**.  

El sistema permite detectar y anonimizar entidades sensibles presentes en textos clínicos no estructurados (historias clínicas, evolutivos, informes médicos), cumpliendo con los principios de minimización y protección de datos establecidos por el RGPD y la legislación española vigente.

El núcleo del proyecto se apoya en:
- Modelos **NER (Named Entity Recognition)** entrenados con **spaCy**
- El corpus **MEDDOCAN** como base de entrenamiento
- Extensiones específicas para el contexto clínico español (p. ej. fármacos)

---

## 2. Objetivo del proyecto

El objetivo principal del TFM es:

> Diseñar, entrenar y validar un sistema reproducible de anonimización automática de texto clínico en español, capaz de sustituir o complementar los procesos manuales tradicionales, reduciendo tiempo, errores humanos y riesgo de exposición de datos sensibles.

Objetivos secundarios:
- Analizar el rendimiento de modelos NER aplicados a texto clínico real
- Proponer una arquitectura reutilizable en otros proyectos de investigación
- Garantizar la reproducibilidad técnica y ética del sistema

---

## 3. Arquitectura del sistema

El sistema se estructura en tres capas claramente diferenciadas:

1. **Capa de inferencia (core del TFM)**  
   - Carga del modelo entrenado  
   - Procesamiento de texto libre  
   - Sustitución controlada de entidades sensibles  

2. **Capa de entrenamiento (metodológica)**  
   - Conversión de datos anotados (BRAT → spaCy)
   - Entrenamiento supervisado del modelo NER
   - Evaluación del rendimiento (precisión, recall, F1)

3. **Capa de demostración y reproducibilidad**  
   - Notebook en Google Colab
   - Ejemplos sintéticos sin datos reales
   - Ejecución completa en entorno aislado

---

## 4. Estructura del repositorio:

anonim-meddocan/
│
├── README.md ← Documento principal (este archivo)
├── requirements.txt ← Dependencias del proyecto
├── config.cfg ← Configuración del pipeline spaCy
│
├── src/ ← Núcleo funcional del sistema
│ ├── anonymize_inference.py ← Script principal de anonimización
│ └── utils.py ← Funciones auxiliares
│
├── models/
│ └── model-best/ ← Modelo final entrenado (spaCy)
│
├── notebooks/
│ └── ANONIM_TFM_Demo.ipynb ← Notebook demostrativo reproducible
│
├── training/ ← Metodología de entrenamiento
│ ├── train_meddocan_safe.py
│ ├── convert_brat_to_spacy.py
│ ├── evaluate_ner.py
│ └── README_TRAINING.md
│
├── data/
│ ├── sample/ ← Textos de ejemplo (sintéticos)
│ └── README_DATASETS.md
│
└── docs/
└── figuras y esquemas del sistema

---


---

## 5. Dataset y consideraciones éticas

- El modelo ha sido entrenado utilizando el dataset MEDDOCAN, un corpus público y anonimizado.
- No se incluyen datos clínicos reales en este repositorio.
- Los ejemplos disponibles son sintéticos o simulados.
- El desarrollo y uso del sistema se enmarca en un proyecto aprobado por un Comité de Ética en Investigación con Medicamentos (CEIm).

En ningún momento se han subido datos clínicos reales a plataformas externas ni a servicios de terceros.

---

## 6. Reproducibilidad

El proyecto es totalmente reproducible mediante:

- Clonado del repositorio
- Instalación de dependencias (`requirements.txt`)
- Ejecución del notebook `ANONIM_Clinico.ipynb` en Google Colab

El notebook demuestra:
- Carga del modelo entrenado
- Anonimización de textos de ejemplo
- Resultados antes/después del proceso

No es necesario reentrenar el modelo para validar el funcionamiento del sistema.

---

## 7. Resultados y evaluación

El modelo final alcanza métricas elevadas en el conjunto de validación del corpus MEDDOCAN, con valores de **precisión, recall y F1 superiores al 0.95** en la mayoría de las entidades clínicas evaluadas.

Los resultados detallados del entrenamiento y evaluación se documentan en la memoria del TFM y en la carpeta `training/`.

---

## 8. Aportación del Trabajo Fin de Máster

La contribución original de este TFM incluye:

- Integración práctica de modelos NER en un flujo clínico realista
- Adaptación del modelo a texto clínico en español
- Arquitectura reutilizable para investigación clínica
- Enfoque explícito en anonimización ética y reproducible
- Puente entre investigación académica y aplicación clínica real

---

## 9. Licencia

Este proyecto se distribuye bajo licencia **MIT**, permitiendo su reutilización académica y científica con atribución adecuada.



Enlace al Colab:
https://colab.research.google.com/drive/1RLlUBuLDNCC3J5sslut8Jt-reckdXmPl?usp=sharing


---

## Autor

César Carballo Cardona
Trabajo Fin de Máster – Máster en aplicaciones de la Inteligencia Artifial en la sanidad.
CENTRO EUROPEO DE MÁSTERES Y POSGRADOS

