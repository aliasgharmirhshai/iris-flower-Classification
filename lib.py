from typing import List, NamedTuple, TypeVar, Tuple, List
from collections import Counter
import random
from math import sqrt

# linear algebra
Vector = List[float]

def distance(v: Vector, w: Vector) -> float:
    """Computes the Euclidean distance between two vectors v and w."""
    assert len(v) == len(w), "vectors must be same length"

    squared_diffs = [(v_i - w_i) ** 2 for v_i, w_i in zip(v, w)]
    sum_of_squares = sum(squared_diffs)
    return sqrt(sum_of_squares)

# spilit data 
X = TypeVar('X')
Y = TypeVar('Y')

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * prob)
    return data[:cut], data[cut:]

def train_test_split(xs: List[X],
                     ys: List[Y],
                     test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs], # x_train
            [xs[i] for i in test_idxs],  # x_test
            [ys[i] for i in train_idxs], # y_train
            [ys[i] for i in test_idxs])  # y_test


# knn
def raw_majority_vote(labels: List[str]) -> str:
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

def majority_vote(labels: List[str]) -> str:
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winner = len([count 
                    for count in vote_counts.values()
                    if count == winner_count])
    if num_winner == 1:
        return winner
    else:
        return majority_vote(labels[:-1])

class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classify(k: int,
                labeled_points: List[LabeledPoint],
                new_point: Vector) -> str:
    
    # Order the labeled points from nearest to farthest.
    by_distance = sorted(labeled_points,
                        key=lambda lp: distance(lp.point, new_point))
    
    # Find the labels for the k closest
    k_nearest_labels = [lp.label for lp in by_distance[:k]]

    return majority_vote(k_nearest_labels)

