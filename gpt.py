#!/usr/bin/env python3

import atexit
import code
import json
import openai
import os
import readline
import rlcompleter
import tiktoken

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
    def prep_source_base(cls, dir, exclude_dirs=["artifacts", "node_modules"]):
        file_extensions = [".py", ".c", ".cpp", ".js", ".java", ".cs", ".go", ".rb", ".php", ".swift", ".ts"]
        count, size, token_count = 0, 0, 0
        msgs = []
        enc = tiktoken.get_encoding("cl100k_base")
        for root, dirs, files in os.walk(dir):
            for file in files:
                if any(s in root for s in exclude_dirs):
                    continue
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        file_content = f.read()
                        tokens = enc.encode(file_content)
                    msgs.append({"role": "user", "content": f"{file_path}\n\n{file_content}"})
                    count, size, token_count  = count + 1, size + len(file_content), token_count + len(tokens)
        return {"count": count, "size": size, "token_count": token_count, "messages": msgs}
    
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
