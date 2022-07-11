import collaborative_filtering as cf
import numpy as np

item1 = np.array([5, 6, 7, 8])
item2 = np.array([5, 7, 6, 7])

print('Using average difference:')
print(cf.calculate_similarity_avg_diff(item1, item2))
print('------------------------')
print('Using cosine similarity:')
print(cf.calculate_similarity_avg_diff(item1, item2))
