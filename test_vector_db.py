# test_vector_db.py
#
# STRESS_TEST=1 python -m unittest test_vector_db.py

import hashlib
import os
import shutil
import time
import unittest
from vector_db import ChromaDB

class TestChromaDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_name = 'test.db'
        cls.collection_name = 'test'
        cls.chroma_db = ChromaDB(path=cls.db_name, name=cls.collection_name)

    @classmethod    
    def tearDownClass(cls):
        if os.path.exists(cls.db_name):
            shutil.rmtree(cls.db_name)
    
    def test_add(cls):
        cls.chroma_db.client.delete_collection(cls.collection_name)
        cls.chroma_db.collection = cls.chroma_db.client.create_collection(cls.collection_name)

        # embedding & id
        embedding_1 = [1.1, 2.1, 3.9, 4.3]
        id_1 = 'id_1'

        # embedding, id & metadata
        embedding_2 = [1.2, 2.1, 3.9, 4.3]
        id_2 = 'id_2'
        metadata_2 = {'name': id_2}
        cls.chroma_db.add([id_1, id_2], [embedding_1, embedding_2], [None, metadata_2])

        # try to add them again, should succeed with printing warnings
        cls.chroma_db.add([id_1, id_2], [embedding_1, embedding_2], [None, metadata_2])

        # exception with different embedding length
        embedding_3 = [2.2, 2.1, 3.9, 4.3, 5.1]
        id_3 = 'id_3'
        with cls.assertRaises(Exception):
            cls.chroma_db.add([id_3], [embedding_3], [None])

        # this time should succeed
        embedding_3 = embedding_3[:-1]
        cls.chroma_db.add_one(id_3, embedding_3, None)

        # query test
        embedding_3 = [2.2, 2.1, 3.9, 4.3]
        data = cls.chroma_db.query(embedding_3, n_results=2)
        cls.assertEqual(['id_3', 'id_2'], data['ids'][0])

        embedding = [1, 1, 1, 1]
        data = cls.chroma_db.query(embedding, n_results=3)
        cls.assertEqual(['id_1', 'id_2', 'id_3'], data['ids'][0])

        embedding = [100, 100, 100, 100]
        data = cls.chroma_db.query(embedding, n_results=3)
        cls.assertEqual(['id_3', 'id_2', 'id_1'], data['ids'][0])
        return
    
    def test_default_embedding(cls):
        cls.chroma_db.client.delete_collection(cls.collection_name)
        cls.chroma_db.collection = cls.chroma_db.client.create_collection(cls.collection_name)

        # embedding & id
        id_1 = 'id_1'
        data_1 = id_1

        # embedding, id & metadata
        id_2 = 'id_2'
        data_2 = id_2
        metadata_2 = {'name': id_2}
        cls.chroma_db.collection.add(
            documents=[data_1, data_2],
            ids=[id_1, id_2],
            metadatas=[None, metadata_2]
        )

        # try to add them again, should succeed with printing warnings
        cls.chroma_db.collection.add(
            documents=[data_1, data_2],
            ids=[id_1, id_2],
            metadatas=[None, metadata_2]
        )

        # exception with different embedding length
        embedding_3 = [2.2, 2.1, 3.9, 4.3, 5.1]
        id_3 = 'id_3'
        data_3 = id_3
        with cls.assertRaises(Exception):
            cls.chroma_db.collection.add(
                documents=[data_3],
                embeddings=[embedding_3],
                ids=[id_3],
                metadatas=[None]
            )

        # this time should succeed
        cls.chroma_db.collection.add(
            documents=[data_3],
            ids=[id_3],
            metadatas=[None]
        )

        # query test
        embedding_3 = [1] * 384
        data = cls.chroma_db.query(embedding_3, n_results=2)
        cls.assertEqual(['id_3', 'id_2'], data['ids'][0])

        embedding = [2] * 384
        data = cls.chroma_db.query(embedding, n_results=3)
        cls.assertEqual(['id_3', 'id_2', 'id_1'], data['ids'][0])

        embedding = [100] * 384
        data = cls.chroma_db.query(embedding, n_results=3)
        cls.assertEqual(['id_3', 'id_2', 'id_1'], data['ids'][0])
        return

    @unittest.skipUnless(os.getenv('STRESS_TEST') == '1', 'skip stress test')
    def test_stress(cls):
        cls.chroma_db.client.delete_collection(cls.collection_name)
        cls.chroma_db.collection = cls.chroma_db.client.create_collection(cls.collection_name)

        start_time = time.time()
        embedding_dimension = 1600
        count = 10000
        for i in range(count):
            id = hashlib.sha256(str(i).encode()).hexdigest()
            embedding = [i] * embedding_dimension
            cls.chroma_db.add_one(id, embedding, None)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Adding {count} took {format(duration, '.2f')} seconds, {format(count / duration, '.2f')} items per second")

        query_count = 1000
        start_time = time.time()
        for i in range(query_count):
            data = cls.chroma_db.query([i] * embedding_dimension, n_results=10)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Querying {query_count} took {format(duration, '.2f')} seconds, {format(query_count / duration, '.2f')} items per second")
    
if __name__ == '__main__':
    unittest.main()

# EoF