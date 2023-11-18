# test_embedding.py
#
# STRESS_TEST=1 python -m unittest test_embedding.py

import os
import unittest
from gpt import GPT

class TestEmbedding(unittest.TestCase):
    def test_embedding(self):
        return

    # test case for GPT.text_to_chunks
    def test_text_to_chunks(self):
        # Test 1: Normal case
        text = "This is a test text for the text_to_chunks method."
        chunks = GPT.text_to_chunks(text, chunk_size=10, overlap_size=5)
        self.assertEqual(chunks, [(0, 10, 'This is a '), (5, 15, 'is a test '), (10, 23, 'test text for'), (20, 30, 'for the te'), (28, 38, 'text_to_ch'), (33, 43, 'to_chunks '), (38, 50, 'unks method.')])

        # Test 2: chunk_size is larger than the length of the text
        text = "Short text."
        chunks = GPT.text_to_chunks(text, chunk_size=50, overlap_size=5)
        self.assertEqual(chunks, [(0, 11, "Short text.")])

        # Test 3: overlap_size is larger than the length of the text
        text = "Short text."
        chunks = GPT.text_to_chunks(text, chunk_size=50, overlap_size=50)
        self.assertEqual(chunks, [(0, 11, "Short text.")])

        # Test 4: chunk_size is less than 10
        text = "This is a test text for the text_to_chunks method."
        with self.assertRaises(ValueError):
            GPT.text_to_chunks(text, chunk_size=5, overlap_size=5)

        # Test 5: chunk_size is less than overlap_size
        text = "This is a test text for the text_to_chunks method."
        with self.assertRaises(ValueError):
            GPT.text_to_chunks(text, chunk_size=10, overlap_size=15)

        # Test 6: all non-whitespaces
        text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        chunks = GPT.text_to_chunks(text, chunk_size=10, overlap_size=5)
        self.assertEqual(chunks, [(0, 10, 'abcdefghij'), (5, 15, 'fghijklmno'), (10, 20, 'klmnopqrst'), (15, 25, 'pqrstuvwxy'), (20, 30, 'uvwxyzABCD'), (25, 35, 'zABCDEFGHI'), (30, 40, 'EFGHIJKLMN'), (35, 45, 'JKLMNOPQRS'), (40, 50, 'OPQRSTUVWX'), (45, 55, 'TUVWXYZ012'), (50, 62, 'YZ0123456789')])

        # Test 7: all whitespaces
        text = " " * 50
        chunks = GPT.text_to_chunks(text, chunk_size=10, overlap_size=5)
        self.assertEqual(chunks, [(0, 10, '          '), (5, 15, '          '), (10, 20, '          '), (15, 25, '          '), (20, 30, '          '), (25, 35, '          '), (30, 40, '          '), (35, 50, '               ')])

    @unittest.skipUnless(os.getenv('STRESS_TEST') == '1', 'skip stress test')
    def test_gpt_py(self):
        # Test 8: real file
        with open('gpt.py', 'r') as file:
            text = file.read()
        chunks = GPT.text_to_chunks(text)
        for i in chunks:
            print(i[0], i[1], i[2])

if __name__ == '__main__':
    unittest.main()

# EoF