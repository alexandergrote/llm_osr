

from typing import List
from copy import copy
from pydantic import BaseModel

class LogProb(BaseModel):
    text: str
    logprob: float


def get_target_word(prior_sequence: List[str], log_sequences: List[LogProb]):

    pseq = copy(prior_sequence)

    assert len(prior_sequence) <= len(log_sequences)

    for logprob in log_sequences:

        if len(pseq) < 1:
            return logprob.text
        
        print(pseq)
        print(logprob.text, pseq[0], logprob.text == pseq[0])

        if logprob.text == pseq[0]:
            pseq.pop(0)
        else:
            pseq = copy(prior_sequence)
            

if __name__ == '__main__':

    prior_sequence = ['alex', 'hello', 'bye']
    log_sequences = [
        LogProb(text='alex', logprob=0),
        LogProb(text='hello', logprob=0), 
        LogProb(text='my', logprob=1),
        LogProb(text='bye', logprob=0)
    ]

    print(get_target_word(prior_sequence=prior_sequence, log_sequences=log_sequences))


