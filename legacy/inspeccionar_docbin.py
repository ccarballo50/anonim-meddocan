import spacy
from spacy.tokens import DocBin

db = DocBin().from_disk("WEB meddocan/train/train.spacy")
docs = list(db.get_docs(spacy.blank("es").vocab))
print("Número de documentos:", len(docs))
