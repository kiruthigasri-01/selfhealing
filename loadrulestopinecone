from google import genai
from pinecone import Pinecone
import re

GEMINI_API_KEY = ""
PINECONE_API_KEY = ""

client = genai.Client(api_key=GEMINI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("business-rules")

with open("business_rules.txt","r",encoding="utf-8") as f:
    content = f.read()

rules = re.split(r"Rule ID:", content)

vectors = []

for rule in rules:

    rule = rule.strip()

    if not rule:
        continue

    lines = rule.split("\n")

    rule_id = lines[0].strip()

    rule_text = " ".join(lines[1:]).strip()

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=rule_text
    )

    vector = response.embeddings[0].values

    vectors.append({
        "id": rule_id,
        "values": vector,
        "metadata": {
            "rule_text": rule_text
        }
    })

index.upsert(
    vectors=vectors
)

print(f"Loaded {len(vectors)} rules")
