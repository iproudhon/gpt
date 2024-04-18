# vector-db.py
#
# pip install pysqlite3-binary
# pip install chromadb

import chromadb
import os

class ChromaDB:
    def __init__(self, path, name, distance_space='cosine'):
        if not distance_space in ["l2", "ip", "cosine"]:
            raise ValueError("distance_space must be one of 'l2', 'ip', 'cosine'")
        self.path = path
        self.client = chromadb.PersistentClient(path=path)
        metadata = {"hnsw:space": distance_space}
        self.collection = self.client.get_or_create_collection(name=name, metadata=metadata)

    def add(self, ids, embeddings=None, metadatas=None, documents=None):
        return self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
    
    def add_one(self, id, embedding=None, metadata=None, document=None):
        embeddings = embedding and [embedding] or None
        metadatas = metadata and [metadata] or None
        documents = document and [document] or None
        return self.add([id], embeddings=embeddings, metadatas=metadatas, documents=documents)
    
    def query(self, embedding, n_results=10):
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )

    # static test methods
    @staticmethod
    def load_file(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
        return content

    @staticmethod
    def get_openai_key():
        api_key_path = os.path.expanduser("~/.openai_api_key")
        key = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"] or (os.path.isfile(api_key_path) and ChromaDB.load_file(api_key_path).strip())
        return key

    @staticmethod
    def get_db(path, name, distance_space='cosine'):
        return ChromaDB(path=path, name=name, distance_space=distance_space)

    @staticmethod
    def get_openai_embedding_func():
        api_key = ChromaDB.get_openai_key()
        model_name = "text-embedding-ada-002"
        return chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=model_name)

    @staticmethod
    def get_default_embedding_func():
        return chromadb.utils.embedding_functions.DefaultEmbeddingFunction()

# EoF
