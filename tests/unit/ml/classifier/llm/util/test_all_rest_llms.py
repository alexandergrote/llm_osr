import unittest

from src.ml.classifier.llm.util.rest import StructuredRequestLLM
from src.util.constants import Directory
from src.util.dynamic_import import DynamicImport
from src.ml.classifier.llm.util.logprob import LogProbScore
from src.ml.classifier.llm.util.prediction import PredictionV1

prompt = """
    Please classify this sentence: "What is the meaning of life?"

    You must comply will the following instructions:

    - There are two possible labels and you have to decide which one to use: "question" and "answer".
    - Fill in your label and your reasoning using this format:
        {{"reasoning": "<your reasoning>", "label": "<your label>"}}  
    - Reply only with valid json
    """

PredictionV1.valid_labels = ["question", "answer"]


class TestLLM(unittest.TestCase):

    def setUp(self):
        
        yaml_dir = Directory.CONFIG / 'llm'
        self.yaml_files = yaml_dir.glob("*.yaml")

    def test_llm(self):

        for yaml_file in self.yaml_files:

            llm = DynamicImport.init_class_from_yaml(
                filename=yaml_file
            )

            with self.subTest(msg=yaml_file):

                self.assertTrue(isinstance(llm, StructuredRequestLLM))

                print(yaml_file)

                output = llm(text=prompt, pydantic_model=PredictionV1, use_cache=True)

                self.assertTrue(isinstance(output, LogProbScore))



if __name__ == '__main__':
    unittest.main()



