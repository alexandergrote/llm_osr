"""Implement semantic entropy."""
import logging
import os

import numpy as np

from abc import abstractmethod
from enum import Enum
from pydantic import BaseModel
from typing import List, Type
from tenacity import retry, wait_random_exponential, retry_if_not_exception_type

from openai import OpenAI


CLIENT = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', False))


class KeyError(Exception):
    """OpenAIKey not provided in environment variable."""
    pass


@retry(retry=retry_if_not_exception_type(KeyError), wait=wait_random_exponential(min=1, max=10))
def predict(prompt, temperature=1.0, model='gpt-4'):
    """

    Predict with GPT models.
    
    """

    if not CLIENT.api_key:
        raise KeyError('Need to provide OpenAI API key in environment variable `OPENAI_API_KEY`.')

    if isinstance(prompt, str):
        messages = [
            {'role': 'user', 'content': prompt},
        ]
    else:
        messages = prompt

    if model == 'gpt-4':
        model = 'gpt-4-0613'
    elif model == 'gpt-4-turbo':
        model = 'gpt-4-1106-preview'
    elif model == 'gpt-3.5':
        model = 'gpt-3.5-turbo-1106'

    output = CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=200,
        temperature=temperature,
    )

    response = output.choices[0].message.content
    
    return response


class EntailmentEnum(int, Enum):
    """Enum for entailment."""
    NEUTRAL = 1
    ENTAILMENT = 2
    CONTRADICTION = 0

    @classmethod
    def get_all_values(cls) -> List[int]:
        return [value.value for value in cls]


class BaseEntailmentModel(BaseModel):

    @abstractmethod
    def predict(self, text1, text2, question, temperature) -> EntailmentEnum:
        raise NotImplementedError("Not implemented")
    
    def are_equivalent(self, text1: str, text2: str, question: str, strict_entailment: bool) -> bool:

        implication_1 = self.predict(text1, text2, question=question, temperature=1)
        implication_2 = self.predict(text2, text1, question=question, temperature=1)

        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            return (implication_1 == EntailmentEnum.ENTAILMENT) and (implication_2 == EntailmentEnum.ENTAILMENT)

       
        implications = [implication_1, implication_2]

        # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
        return (EntailmentEnum.CONTRADICTION not in implications) and ([EntailmentEnum.NEUTRAL, EntailmentEnum.NEUTRAL] != implications)
    

class EntailmentLLMMixin:

    @staticmethod
    def equivalence_prompt(text1: str, text2: str, question: str) -> str:

        prompt = f"We are evaluating answers to the question \"{question}\"\n"
        prompt += "Here are two possible answers:\n"
        prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
        prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond with entailment, contradiction, or neutral."

        return prompt
    
    @staticmethod
    def parse_ouput(response: str) -> EntailmentEnum:

        binary_response = response.lower()[:30]

        if EntailmentEnum.ENTAILMENT.name.lower() in binary_response:
            return EntailmentEnum.ENTAILMENT
        elif EntailmentEnum.NEUTRAL.name.lower() in binary_response:
            return EntailmentEnum.NEUTRAL
        elif EntailmentEnum.CONTRADICTION.name.lower() in binary_response:
            return EntailmentEnum.CONTRADICTION
        else:
            logging.warning('MANUAL NEUTRAL!')
            return EntailmentEnum.NEUTRAL 


class EntailmentLLM(EntailmentLLMMixin, BaseEntailmentModel):

    def predict(self, text1, text2, question, temperature) -> EntailmentEnum:

        prompt = EntailmentLLM.equivalence_prompt(text1, text2, question)

        response = predict(prompt, temperature, model='gpt-4')
        
        return self.parse_ouput(response)
        
    
class SemanticUncertainty(BaseModel):

    model: Type[BaseEntailmentModel] = EntailmentLLM()
    strict_entailment: bool = False

    def get_semantic_ids(self, strings_list, example: str) -> List[int]:
    
        """
        Group list of predictions into semantic meaning.
        """

        # Initialise all ids with -1.
        semantic_set_ids = [-1] * len(strings_list)

        # Keep track of current id.
        # start from zero and increment by 1 if string1
        next_id = 0

        for i, string1 in enumerate(strings_list):

            # Check if string1 already has an id assigned.
            if semantic_set_ids[i] == -1:

                # If string1 has not been assigned an id, assign it next_id.
                semantic_set_ids[i] = next_id

                for j in range(i+1, len(strings_list)):

                    # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                    if self.model.are_equivalent(text1=string1, text2=strings_list[j], question=example, strict_entailment=self.strict_entailment):  # type: ignore
                        semantic_set_ids[j] = next_id

                # increment next_id by 1.
                next_id += 1

        assert -1 not in semantic_set_ids

        return semantic_set_ids

    @staticmethod
    def cluster_assignment_entropy(semantic_ids: List[int]):
        """
        Estimate semantic uncertainty from how often different clusters get assigned.

        We estimate the categorical distribution over cluster assignments from the
        semantic ids. The uncertainty is then given by the entropy of that
        distribution. This estimate does not use token likelihoods, it relies soley
        on the cluster assignments. If probability mass is spread of between many
        clusters, entropy is larger. If probability mass is concentrated on a few
        clusters, entropy is small.

        Input:
            semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
        Output:
            cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
        """

        n_generations = len(semantic_ids)
        counts = np.bincount(semantic_ids)

        probabilities = counts / n_generations
        assert np.isclose(probabilities.sum(), 1)

        return - (probabilities * np.log(probabilities)).sum()

    def get_score(self, responses: List[str], question: str) -> float:

        semantic_ids = self.get_semantic_ids(strings_list=responses, example=question)

        return self.cluster_assignment_entropy(semantic_ids)


def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum_normalized'):
    """
    Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))

    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        # Gather log likelihoods at these indices.
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum_normalized':
            # log_lik_norm = id_log_likelihoods - np.prod(log_likelihoods)
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


def predictive_entropy(log_probs):
    """
    Compute MC estimate of entropy.

    `E[-log p(x)] ~= -1/N sum_i log p(x_i)`, i.e. the average token likelihood.
    """

    entropy = -np.sum(log_probs) / len(log_probs)

    return entropy


def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def cluster_assignment_entropy(semantic_ids):
    """
    Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)

    probabilities = counts / n_generations
    assert np.isclose(probabilities.sum(), 1)

    return - (probabilities * np.log(probabilities)).sum()

    """
    # Compute entropy from frequencies of cluster assignments.
    entropies['cluster_assignment_entropy'].append(cluster_assignment_entropy(semantic_ids))

    # Length normalization of generation probabilities.
    log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

    # Compute naive entropy.
    entropies['regular_entropy'].append(predictive_entropy(log_liks_agg))

    # Compute semantic entropy.
    log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
    pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
    entropies['semantic_entropy'].append(pe)
    """
    

if __name__ == '__main__':

    """
    from sklearn.metrics import roc_curve, auc

    y_true = np.array([0, 1, 1, 0, 1])
    y_score = np.array([1.5, 0.5, 2.5, 2, 2.3])

    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    print(thresholds)
    """

    semantic_ids = [0, 1, 2, 1]
    log_liks = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]
    log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

    entropy = cluster_assignment_entropy(semantic_ids=semantic_ids)
    print(entropy)

    # probability of class prediction is derived from entailment and reasoning
    # e.g. prediction is class 1 due to majority vote, then probability of class 1 is 1/2

    log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
    pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
    print(pe)

