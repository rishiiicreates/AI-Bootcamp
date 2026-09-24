import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import spacy

# Load spaCy's lightweight English core model (~12MB)
nlp = spacy.load("en_core_web_sm")

# Sample unstructured text input
raw_resume_text = """Hrishikesh Yadav is a Software Developer and B.Tech CSE student living in Ghaziabad. He has worked as an AI intern at InAmigos Foundation and developed platforms like DashMetrics, Housel, and Nudge. Hrishikesh studies at SRM Institute of Science and Technology and has expertise in Python, Full-Stack Web Development, and AI Data Analytics."""

print("Processing text with spaCy...\n")
doc = nlp(raw_resume_text)

print("--- Extracted Named Entities ---")
# Extract entities like Persons, Organizations, Locations, and Dates
for ent in doc.ents:
  print(f"Entity: {ent.text:<20} | Type: {ent.label_:<10} ({spacy.explain(ent.label_)})")

print("\n--- Identified Keywords/Nouns ---")
# Extract key nouns/skills
nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
print("Key Nouns Found:", list(set(nouns)))
