#!/usr/bin/env python3

import atexit
import code
import fnmatch
import hashlib
import magic
import openai
import os
import readline
import shutil
import tiktoken

import chromadb
from vector_db import ChromaDB

class GPT:
    # chat models
    #   gpt-4, gpt-4-32k
    #   gpt-4-1106-preview, a.k.a. gpt-4-turbo
    #   gpt-4-1106-vision-preview
    # embedding model
    #   text-embedding-ada-002 
    def __init__(self, model="gpt-4-1106-preview", embedding_model="text-embedding-ada-002"):
        self.model = model
        self.embedding_model = embedding_model
        api_key_path = os.path.expanduser("~/.openai_api_key")
        key = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"] or (os.path.isfile(api_key_path) and self.load_file(api_key_path).strip())
        self.client = openai.OpenAI(api_key=key)
        del key, api_key_path
        self.messages = [
            {"role": "system", "content": "You're an expert coder and a sharp critic. If you don't know, don't make up, just say you don't know."},
        ]

    @staticmethod
    def load_file(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
        return content

    @staticmethod
    def load_script(file_path):
        with open(file_path, 'r') as file:
            script = file.read()
        exec(script, globals())

    def send(self, msg, stream=True):
        self.messages.append({"role": "user", "content": msg})
        if not stream:
            reply = self.client.chat.completions.create(
                model=self.model, messages=self.messages
            )
            response_message = reply.choices[0].message
            self.messages.append(response_message)
            return response_message.content
        else:
            reply = self.client.chat.completions.create(
                model=self.model, messages=self.messages, stream=True
            )
            content = ""
            for chunk in reply:
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    content += delta.content
                    print(delta.content, end='', flush=True)
            print()
            self.messages.append({"role": "assistant", "content": content})
            return content
        
    def get_embeddings(self, texts):
        embeddings = self.client.embeddings.create(input=texts, model=self.embedding_model)
        return [item.embedding for item in embeddings.data]

    def get_embedding(self, text):
        return self.get_embeddings([text])[0]

    @classmethod
    def text_to_chunks(cls, text, chunk_size=500, overlap_size=100):
        if chunk_size < overlap_size or chunk_size < 10 or overlap_size < 0:
            raise ValueError("chunk_size must be at least 10 and greater than overlap_size")

        overfill_size = overlap_size > 0 and int(overlap_size / 2) or int(chunk_size / 2)
        chunks = []
        start = 0
        while start < len(text):
            # Find the end of the chunk
            end = start + chunk_size
            if end > len(text) or len(text) - end <= overlap_size:
                end = len(text)
            else:
                # Make sure to end the chunk at a space, extending the chunk within overfill_size if necessary
                next = end
                while next < len(text) and next - end <= overfill_size and not text[next].isspace():
                    next += 1
                if text[next].isspace():
                    end = next
            chunk = text[start:end]
            chunks.append((start, end, chunk))

            if end >= len(text):
                break

            # Move the next start to the end minus the overlap size
            # Make sure to start at a non-whitespace character, shriniking the chunk within overfill_size if necessary
            start = end - overlap_size
            next = max(start, 1)
            while next < end - overfill_size and not (text[next-1].isspace() and not text[next].isspace()):
                next += 1
            if text[next-1].isspace() and not text[next].isspace():
                start = next
        return chunks
    
    # prepare [{ "role": "user", "content": "<file-name>\n\n<file-content>" }]
    # from source files under dir/*/*
    @classmethod
    def prep_source_base(cls, dir, excludes=["artifacts", "node_modules", ".git", ".gitignore", "__pycache__"]):
        file_extensions = [".py", ".c", ".cpp", ".js", ".java", ".cs", ".go", ".rb", ".php", ".swift", ".ts", ".sol"]
        count, size, token_count = 0, 0, 0
        msgs = []
        enc = tiktoken.get_encoding("cl100k_base")
        for root, dirs, files in os.walk(dir):
            for file in files:
                if any(s in root for s in excludes):
                    continue
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        file_content = f.read()
                        tokens = enc.encode(file_content)
                    msgs.append({"role": "user", "content": f"{file_path}\n\n{file_content}"})
                    count, size, token_count  = count + 1, size + len(file_content), token_count + len(tokens)
        return {"count": count, "size": size, "token_count": token_count, "messages": msgs}

    @staticmethod
    def get_openai_key():
        api_key_path = os.path.expanduser("~/.openai_api_key")
        key = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"] or (os.path.isfile(api_key_path) and self.load_file(api_key_path).strip())
        return key

    @staticmethod
    def get_embedding_db(path, collection_name):
        embedding_db = ChromaDB(path=path, name=collection_name)
        return embedding_db

    @staticmethod
    def get_openai_embedding_func():
        embedding_func = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=model_name)
        return embedding_func

    @staticmethod
    def get_chroma_embedding_func():
        embedding_func = chromadb.utils.embedding_functions.ChromaEmbeddingFunction(embedding_db=embedding_db)
        return embedding_func

    @classmethod
    def embed_files(cls, dir, db, embedding_func, excludes=["artifacts", "node_modules", ".git", ".gitignore", "__pycache__", "*.db"]):
        magic_obj = magic.Magic(mime=True)
        max_size = 100000
        chunk_size = 500
        overlap_size = 100
        for root, dirs, files in os.walk(dir):
            dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pattern) for pattern in excludes)]
            for file in files:
                if any(fnmatch.fnmatch(file, pattern) for pattern in excludes):
                    continue
                file_path = os.path.join(root, file)
                file_type = magic_obj.from_file(file_path)
                if not file_type.startswith('text'):
                    continue
                print(file_path, '->', file_type)
                chunks = []
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    if len(file_content) >= max_size:
                        continue
                    chunks = cls.text_to_chunks(file_content, chunk_size=chunk_size, overlap_size=overlap_size)
                except UnicodeDecodeError as e:
                    print(file_path, e)
                    continue
                if len(chunks) == 0:
                    continue
                ids, embeddings, docs, metadatas = [], [], [], []
                for chunk in chunks:
                    id = f"{file_path}:{chunk[0]}:{chunk[1]}:{chunk[2]}"
                    id = hashlib.sha256(id.encode()).hexdigest()
                    embedding = embedding_func([chunk[2]])[0]
                    ids.append(id)
                    embeddings.append(embedding)
                    docs.append(chunk[2])
                    metadatas.append({"file": file_path, "start": chunk[0], "end": chunk[1]})
                db.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)

    @staticmethod
    def setup_embed_test():
        path = "/mnt/tmp/test.db"
        collection_name = "test"
        distance_space = "cosine"
        db = ChromaDB.get_db(path=path, name=collection_name, distance_space=distance_space)
        ef = ChromaDB.get_openai_embedding_func()
        # ef = ChromaDB.get_default_embedding_func()
        return db, ef

    @staticmethod
    def teardown_embed_test():
        path = "/mnt/tmp/test.db"
        if os.path.exists(path):
            shutil.rmtree(path)

    @staticmethod
    def query_embedding(db, ef, text, n_results=10):
        embedding = ef([text])[0]
        data = db.query(embedding, n_results=n_results)
        metadatas = data['metadatas'][0]
        documents = data['documents'][0]
        distances = data['distances'][0]
        for i in range(len(metadatas)):
            metadata = metadatas[i]
            print(f"{i}: {distances[i]:.2f} {metadata['file']}:{metadata['start']}-{metadata['end']}\n{documents[i]}")
        return data
    
def main():
    global gpt
    gpt = GPT()

    # handle readline history
    history_file = os.path.expanduser("~/.python_history")
    if os.path.exists(history_file):
        readline.read_history_file(history_file)
    atexit.register(lambda: readline.write_history_file(history_file))

    # Explicitly set up readline completer if not already set
    if 'libedit' in readline.__doc__:
        readline.parse_and_bind("bind -e")
        readline.parse_and_bind("bind '\t' rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
    readline.parse_and_bind('"\C-p": history-search-backward')
    readline.parse_and_bind('"\e[A": history-search-backward')
    readline.parse_and_bind('"\C-n": history-search-forward')
    readline.parse_and_bind('"\e[B": history-search-forward')
    readline.parse_and_bind('"\C-r": reverse-search-history')

    code.interact(local=globals())

if __name__ == "__main__":
    main()

# EOF
