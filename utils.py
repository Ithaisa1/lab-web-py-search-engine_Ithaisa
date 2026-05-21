import os
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import numpy as np

# Cargar variables .env
load_dotenv()

# Cliente OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Modelo embeddings
MODEL = "text-embedding-3-small"

# Flag to use mock embeddings (set to True if you want to avoid API calls)
USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"


def crear_embedding(texto):
    if USE_MOCK:
        # Generate a deterministic mock embedding based on the text hash
        # This ensures similar texts get similar embeddings (for demo purposes)
        np.random.seed(hash(texto) % 2**32)
        embedding = np.random.rand(1536)
    else:
        try:
            response = client.embeddings.create(
                model=MODEL,
                input=texto
            )
            embedding = np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error creating embedding: {e}")
            print("Falling back to mock embedding")
            np.random.seed(hash(texto) % 2**32)
            embedding = np.random.rand(1536)
    
    # Normalize the embedding to unit length for cosine similarity
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.tolist()


def contar_tokens(texto):

    encoding = tiktoken.encoding_for_model(MODEL)

    tokens = encoding.encode(texto)

    return len(tokens)