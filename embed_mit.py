
import os
import faiss
from dotenv import load_dotenv
import numpy as np
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

EMBEDDING_DEPLOYMENT =os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
data_folder = "data"
chunk_size = 800

def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model=EMBEDDING_DEPLOYMENT
    )
    return response.data[0].embedding

all_chunks = []
all_embeddings = []

for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as f:
            content = f.read()
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            for chunk in chunks:
                embedding = get_embedding(chunk)
                all_chunks.append(chunk)
                all_embeddings.append(embedding)
                print(f"Embedded chunk from {filename}")

# Store in FAISS index
dimension = len(all_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(all_embeddings).astype("float32"))
faiss.write_index(index, "index.faiss")

# Save chunks
with open("texts.txt", "w", encoding="utf-8") as f:
    f.write("\n\n---\n\n".join(all_chunks))

print("Embeddings saved to index.faiss and chunks saved to texts.txt")
