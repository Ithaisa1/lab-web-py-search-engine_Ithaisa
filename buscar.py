import chromadb

from utils import crear_embedding

# Conectar con ChromaDB
client = chromadb.PersistentClient(
    path="./db"
)

# Obtener colección
collection = client.get_collection(
    name="articulos"
)


def buscar(query, n_resultados=3):

    print(f"\nBUSCANDO: {query}\n")

    # Crear embedding de la query
    embedding_query = crear_embedding(query)

    # Buscar documentos similares
    resultados = collection.query(
        query_embeddings=[embedding_query],
        n_results=n_resultados
    )

    # Extraer resultados
    documentos = resultados["documents"][0]
    metadatos = resultados["metadatas"][0]
    distancias = resultados["distances"][0]

    # Mostrar resultados
    for i in range(len(documentos)):

        similitud = 1 - distancias[i]

        # Ensure similarity is between 0 and 1
        similitud = max(0, min(1, similitud))

        print(f"Resultado #{i+1}")
        print(f"Titulo: {metadatos[i]['titulo']}")
        print(f"Score similitud: {similitud:.4f}")
        print(f"Contenido: {documentos[i]}")
        print("-" * 50)


# =========================
# PRUEBAS
# =========================

buscar("como hacer una API en Python")

buscar("diferencias entre frameworks de frontend")

buscar("como funciona la autenticacion en aplicaciones web")

buscar("herramientas para trabajar con modelos de lenguaje")