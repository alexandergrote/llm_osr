import unittest

from src.ml.classifier.llm.util.tokenizer import LlamaTokenizer



class TestLlamaTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = LlamaTokenizer()

    def test_encode(self):

        text = "hello world"
        encoded = self.tokenizer.encode(text)
        self.assertTrue(isinstance(encoded, list))
        self.assertEqual(len(encoded), 3)
        self.assertTrue(all([isinstance(el, int) for el in encoded]))

if __name__ == '__main__':
    unittest.main()

