# vector-db.py
#
# pip install pysqlite3-binary
# pip install chromadb

import chromadb

class ChromaDB:
    def __init__(self, path, name):
        self.path = path
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=name)

    def add(self, ids, embeddings, metadatas):
        return self.collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def add_one(self, id, embedding, metadata):
        return self.add([id], [embedding], [metadata])        
    
    def query(self, embedding, n_results=10):
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )

# EoF