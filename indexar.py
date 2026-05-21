import chromadb

from data.articulos import articulos
from utils import crear_embedding, contar_tokens

# Crear cliente persistente
client = chromadb.PersistentClient(
    path="./db"
)

# Crear colección
collection = client.get_or_create_collection(
    name="articulos"
)

tokens_totales = 0

# Recorrer artículos
for articulo in articulos:

    print(f"\nProcesando: {articulo['titulo']}")

    # Combinar título + contenido
    texto = f"""
    Titulo: {articulo['titulo']}
    Contenido: {articulo['contenido']}
    """

    # Crear embedding
    embedding = crear_embedding(texto)

    # Contar tokens
    tokens = contar_tokens(texto)

    tokens_totales += tokens

    # Guardar en ChromaDB
    collection.add(
        ids=[articulo["id"]],
        embeddings=[embedding],
        documents=[articulo["contenido"]],
        metadatas=[
            {
                "titulo": articulo["titulo"]
            }
        ]
    )

    print("Indexado correctamente")

# Coste aproximado
coste = (tokens_totales / 1_000_000) * 0.02

print("\nRESUMEN")
print(f"Tokens procesados: {tokens_totales}")
print(f"Coste estimado: ${coste:.6f}")