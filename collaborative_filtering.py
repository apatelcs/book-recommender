import numpy as np

# Calculates similarity score using average difference
def calculate_similarity_avg_diff(item1, item2):
    diff = item1 - item2
    return np.mean(diff)

def calculate_similarity_cosine(item1, item2):
    return np.dot(item1, item2) / (np.linalg.norm(item1) * np.linalg.norm(item2))
