import numpy as np

def calculate_similarity_avg_diff(item1: np.array, item2: np.array) -> np.floating:
    '''
    Calculates similarity score using average difference
    '''
    diff = item1 - item2
    return np.mean(diff)


def calculate_similarity_cosine(item1: np.array, item2: np.array) -> np.floating:
    '''
    Calculates cosine similarity between two arrays
    '''
    return np.dot(item1, item2) / (np.linalg.norm(item1) * np.linalg.norm(item2))

# This does not yet work
# TODO: add a pipeline for dataflow and update prediction function accordingly
def pred(usr, item, item_ind, item_mat, rating_mat):
    sim_arr = np.array([calculate_similarity_cosine(item_mat[i], item) for i in range(len(item_mat))])
    alpha = 1 / np.sum(sim_arr)
    term = 0
    for itm in item_mat:
        term += rating_mat[usr, item_ind] * calculate_similarity_cosine(itm, item)

    return alpha * term
