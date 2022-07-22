import numpy as np
import math
import pytest
from collaborative_filtering import *

def test_similarity_avg_diff() -> None:
    '''
    Tests similarity using average difference
    '''
    item1 = np.array([1, 2, 3, 4, 5])
    item2 = np.array([0, 1, 2, 3, 4])

    assert calculate_similarity_avg_diff(item1, item2) == 1


def test_similarity_cosine() -> None:
    '''
    Tests similarity using cosine similarity
    '''
    item1 = np.array([1, 2, 3, 4, 5])
    item2 = np.array([0, 1, 2, 3, 4])
    norm_i1 = math.sqrt((1 + 4 + 9 + 16 + 25))
    norm_i2 = math.sqrt((0 + 1 + 4 + 9 + 16))

    assert calculate_similarity_cosine(item1, item2) == (40 / (norm_i1 * norm_i2))