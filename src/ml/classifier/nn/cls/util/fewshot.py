import torch

def compute_prototypes(support_features: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
    
    """
    Compute class prototypes from support features and labels

    Taken from https://github.com/sicara/easy-few-shot-learning/blob/53200ec56193e4ea204b07492bd315e341466ba4/easyfsl/methods/utils.py#L7

    Args:
        support_features: for each instance in the support set, its feature vector
        support_labels: for each instance in the support set, its label

    Returns:
        for each label of the support set, the average feature vector of instances with this label
    """

    n_way = len(torch.unique(support_labels))
    # Prototype i is the mean of all instances of features corresponding to labels == i
    return torch.cat(
        [
            support_features[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
    )

def compute_outlier_scores(y_pred_proba: torch.Tensor) -> torch.Tensor:

    """
    We define the outlier score as the negative maximum probability of the predicted classes.

    The higher the outlier score, the more likely the sample is an outlier.

    Args:
        y_pred_proba: the predicted probabilities for each class
    Returns:
        the outlier score
    """
    
    return -torch.max(y_pred_proba, dim=1).values