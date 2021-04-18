from typing import Iterable, Tuple


Edge = Tuple[int, int]


def binary_classification_metrics(target: Iterable[Edge], pred: Iterable[Edge]):
    target = set(target)
    pred = set(pred)
    num_expected = len(target)
    num_predicted = len(pred)
    
    tp = len(target & pred)
    fp = len(pred - target)
    fn = len(target - pred)
    # tn = n * n - (tp + fp + fn)
    
    if num_predicted != 0:
        precision = tp / (tp + fp)
    else:
        precision = 1
        
    if num_expected != 0:
        recall = tp / (tp + fn)
    else:
        recall = 1

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return dict(precision=precision, recall=recall, f1=f1,
                num_expected=num_expected, num_predicted=num_predicted)
