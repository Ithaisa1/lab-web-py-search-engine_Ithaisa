from fastapi import FastAPI
from utils import crear_embedding
import chromadb

app = FastAPI(title="Motor de Búsqueda Semántica", description="API para buscar artículos usando embeddings semánticos")

# Conectar con ChromaDB
client = chromadb.PersistentClient(path="./db")
collection = client.get_collection(name="articulos")

@app.get("/")
async def root():
    return {"message": "Motor de Búsqueda Semántica API"}

@app.get("/buscar")
async def buscar_endpoint(q: str, n_resultados: int = 3):
    """
    Busca artículos similares a la consulta proporcionada
    
    Args:
        q (str): La consulta de búsqueda
        n_resultados (int): Número de resultados a devolver (default: 3)
    
    Returns:
        dict: Resultados de la búsqueda con scores de similitud
    """
    # Crear embedding de la query
    embedding_query = crear_embedding(q)
    
    # Buscar documentos similares
    resultados = collection.query(
        query_embeddings=[embedding_query],
        n_results=n_resultados
    )
    
    # Extraer resultados
    documentos = resultados["documents"][0]
    metadatos = resultados["metadatas"][0]
    distancias = resultados["distances"][0]
    
    # Formatear resultados
    resultados_formateados = []
    for i in range(len(documentos)):
        similitud = 1 - distancias[i]
        similitud = max(0, min(1, similitud))  # Asegurar que esté entre 0 y 1
        
        resultados_formateados.append({
            "id": metadatos[i].get("id", ""),
            "titulo": metadatos[i]["titulo"],
            "contenido": documentos[i],
            "similitud": similitud
        })
    
    return {
        "query": q,
        "resultados": resultados_formateados
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)