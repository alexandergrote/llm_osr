from typing import List
from copy import copy
from pydantic import BaseModel

from src.util.types import LogProb

class LogProbExtractor(BaseModel):

    @staticmethod
    def _standardize_string(s: str) -> str:

        # work on copy
        sc = copy(s)

        # remove special characters
        special_chars = [' ', '\n', '\t', '\r', '"', "'"]
        for special_char in special_chars:
            sc = sc.replace(special_char, '')

        # standardize to lowercase
        sc = sc.lower()

        return sc


    @staticmethod
    def _get_index_after_prior_sequence(prior_sequence: List[str], log_sequences: List[LogProb], start_idx: int = 0) -> int:

        pseq = copy(prior_sequence)

        assert len(prior_sequence) <= len(log_sequences)

        for i, logprob in enumerate(log_sequences):

            if i < start_idx:
                continue

            if len(pseq) < 1:
                return i

            logprob_text_standardized = LogProbExtractor._standardize_string(logprob.text)
            prior_sequence_text_standardized = LogProbExtractor._standardize_string(pseq[0])

            if logprob_text_standardized == prior_sequence_text_standardized:
                pseq.pop(0)
            else:
                pseq = copy(prior_sequence)

        raise Exception("Sequence not found")

    @staticmethod
    def get_target_logprobas(prior_sequence: List[str], end_sequence: List[str], log_sequences: List[LogProb]) -> List[LogProb]:

        assert len(prior_sequence) <= len(log_sequences)
        assert len(end_sequence) <= len(log_sequences)
        assert len(prior_sequence) + len(end_sequence) <= len(log_sequences)

        # set start index
        answer_idx_start = 0

        # overwrite for given prior sequence
        if len(prior_sequence) > 0:
            answer_idx_start = LogProbExtractor._get_index_after_prior_sequence(prior_sequence=prior_sequence, log_sequences=log_sequences)
            
        # set end index
        answer_idx_end = len(log_sequences)

        # overwrite for given end sequence
        if len(end_sequence) > 0:
            answer_idx_end = LogProbExtractor._get_index_after_prior_sequence(prior_sequence=end_sequence, log_sequences=log_sequences, start_idx=answer_idx_start) - len(end_sequence)    
        
        return log_sequences[answer_idx_start:answer_idx_end]
    
    @staticmethod
    def get_specific_logprobas(text: str, log_sequences: List[LogProb]) -> List[LogProb]:

        result_list = [el for el in log_sequences if LogProbExtractor._standardize_string(el.text) == LogProbExtractor._standardize_string(text)]

        return result_list
    

if __name__ == '__main__':

    tokens = [
        {'id': 262, 'text': '   ', 'logprob': 0.0, 'special': False},
        {'id': 13688, 'text': ' Example', 'logprob': -0.0024757385, 'special': False},
        {'id': 512, 'text': ':\n', 'logprob': 0.0, 'special': False},
        {'id': 262, 'text': '   ', 'logprob': 0.0, 'special': False},
        {'id': 341, 'text': ' {\n', 'logprob': 0.0, 'special': False},
        {'id': 286, 'text': '       ', 'logprob': 0.0, 'special': False},
        {'id': 364, 'text': " '", 'logprob': 0.0, 'special': False},
        {'id': 9399, 'text': 'answer', 'logprob': 0.0, 'special': False},
        {'id': 1232, 'text': "':", 'logprob': 0.0, 'special': False},
        {'id': 364, 'text': " '", 'logprob': 0.0, 'special': False},
        {'id': 74152, 'text': 'kö', 'logprob': 0.0, 'special': False},
        {'id': 53835, 'text': 'lj', 'logprob': 0.0, 'special': False},
        {'id': 2642, 'text': 'af', 'logprob': 0.0, 'special': False},
        {'id': 78604, 'text': 'dj', 'logprob': 0.0, 'special': False},
        {'id': 10784, 'text': 'kl', 'logprob': 0.0, 'special': False},
        {'id': 3029, 'text': 'ö', 'logprob': 0.0, 'special': False},
        {'id': 300, 'text': 'as', 'logprob': 0.0, 'special': False},
        {'id': 756, 'text': "',\n", 'logprob': 0.0, 'special': False},
        {'id': 286, 'text': '       ', 'logprob': 0.0, 'special': False},
        {'id': 364, 'text': " '", 'logprob': 0.0, 'special': False},
        {'id': 20489, 'text': 'eason', 'logprob': 0.0, 'special': False},
        {'id': 287, 'text': 'ing', 'logprob': 0.0, 'special': False},
        {'id': 1232, 'text': "':", 'logprob': 0.0, 'special': False},
        {'id': 364, 'text': " '", 'logprob': 0.0, 'special': False},
        {'id': 18433, 'text': 'Because', 'logprob': 0.0, 'special': False},
        {'id': 279, 'text': ' the', 'logprob': 0.0, 'special': False},
        {'id': 17571, 'text': ' phrase', 'logprob': -1.66893e-05, 'special': False},
        {'id': 330, 'text': ' "', 'logprob': 0.0, 'special': False},
        {'id': 15339, 'text': 'hello', 'logprob': 0.0, 'special': False},
        {'id': 1, 'text': '"', 'logprob': 0.0, 'special': False},
        {'id': 374, 'text': ' is', 'logprob': 0.0, 'special': False},
        {'id': 264, 'text': ' a', 'logprob': 0.0, 'special': False},
        {'id': 4279, 'text': ' common', 'logprob': 0.0, 'special': False},
        {'id': 43213, 'text': ' greeting', 'logprob': 0.0, 'special': False},
        {'id': 304, 'text': ' in', 'logprob': 0.0, 'special': False},
        {'id': 1690, 'text': ' many', 'logprob': 0.0, 'special': False},
        {'id': 27833, 'text': ' cultures', 'logprob': 0.0, 'special': False},
        {'id': 323, 'text': ' and', 'logprob': -0.6933594, 'special': False},
        {'id': 15823, 'text': ' languages', 'logprob': 0.0, 'special': False},
        {'id': 24314, 'text': ".'\n", 'logprob': 0.0, 'special': False},
        {'id': 262, 'text': '   ', 'logprob': 0.0, 'special': False},
        {'id': 557, 'text': ' }\n\n', 'logprob': 0.0, 'special': False},
        {'id': 262, 'text': '   ', 'logprob': 0.0, 'special': False},
        {'id': 5321, 'text': ' Please', 'logprob': 0.0, 'special': False},
        {'id': 6013, 'text': ' respond', 'logprob': 0.0, 'special': False},
        {'id': 449, 'text': ' with', 'logprob': 0.0, 'special': False},
        {'id': 279, 'text': ' the', 'logprob': -6.198883e-06, 'special': False},
        {'id': 4495, 'text': ' correct', 'logprob': 0.0, 'special': False},
        {'id': 11036, 'text': ' schema', 'logprob': 0.0, 'special': False},
        {'id': 13, 'text': '.', 'logprob': 0.0, 'special': False},
        {'id': 128009, 'text': '<|eot_id|>', 'logprob': 0.0, 'special': True}
    ]

    tokens_logprob_list: List[LogProb] = [LogProb(**token) for token in tokens]

    prior_sequence = ['answer', "':", " '"]
    end_sequence = ["',\n", '       ']
    
    prior_sequence = ['answer', ':', '']
    end_sequence = [","]

    print(
        LogProbExtractor.get_target_logprobas(prior_sequence=prior_sequence, end_sequence=end_sequence, log_sequences=tokens_logprob_list)
    )
