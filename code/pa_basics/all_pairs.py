
# create all pairs


import numpy as np
from itertools import permutations
import concurrent.futures
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

from scipy.spatial.distance import dice, yule, kulsinski, sokalmichener


############################################################
# PART II
# Making pairs from QSAR datasets.


# Function: Generate all possible pairs from a QSAR dataset
# Input: numpy array of a QSAR dataset
# Output: dictionary 
#         key = (drugA_id, drugB_id); value = [y_pair, x_pair1, x_pair2...]


def paired_data(data, with_similarity=False, with_fp=False):
    pairing_tool = PairingDataset(data, with_similarity, with_fp)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(pairing_tool.parallelised_pairing_process, range(pairing_tool.n_combinations))
    return dict(results)


class PairingDataset:
    def __init__(self, data, with_similarity, with_fp):
        self.data = data
        self.with_similarity = with_similarity
        self.with_FP = with_fp

        self.n_samples, self.n_columns = np.shape(data)
        self.permutation_pairs = list(permutations(range(self.n_samples), 2)) + [(a, a) for a in range(self.n_samples)]
        self.n_combinations = len(self.permutation_pairs)

    def parallelised_pairing_process(self, combination_id):
        sid_a, sid_b = self.permutation_pairs[combination_id]
        sample_a = self.data[sid_a: sid_a + 1, :]
        sample_b = self.data[sid_b: sid_b + 1, :]

        pair_ab = pair_2samples(self.n_columns, sample_a, sample_b, self.with_similarity, self.with_FP)
        return (sid_a, sid_b), pair_ab


# Function: Transform the information from two single samples to a pair
# Input:  integer number of columns; 
#         numpy array of Sample A in the shape of (1, ncolumns)
#         numpy array of Sample B in the shape of (1, ncolumns)
# Output: list of Pair AB
#     
#       Note the Rules of pairwise features:
#           x_A = 1 & x_B = 1 -> X_AB = 2
#           x_A = 1 & x_B = 0 -> X_AB = 1
#           x_A = 0 & x_B = 1 -> X_AB = -1
#           x_A = 1 & x_B = 0 -> X_AB = 0

def pair_2samples(n_columns, sample_a, sample_b, similarity=False, orig_fp=False):
    delta_y = sample_a[0, 0] - sample_b[0, 0]
    new_sample = [delta_y]

    for fid in range(1, n_columns):
        f_value_a = sample_a[0, fid]
        f_value_b = sample_b[0, fid]

        if f_value_a == f_value_b:
            assign = f_value_a + f_value_b
            new_sample.append(assign)

        elif f_value_a != f_value_b:
            assign = f_value_a - f_value_b
            new_sample.append(assign)

    if similarity:
        new_sample += similarity_metrics(np.array([sample_a[0, 1:]]), np.array([sample_b[0, 1:]]))

    if orig_fp:
        new_sample += list(sample_a[0, 1:]) + list(sample_b[0, 1:])

    return new_sample


def similarity_metrics(fp1, fp2):
    # OR manhattan_distances(fp1, fp2)[0, 0],
    return [
        jaccard_score(fp1[0], fp2[0]),
        cosine_similarity(fp1, fp2)[0, 0],
        manhattan_distances(fp1, fp2),
        euclidean_distances(fp1, fp2)[0, 0],
        dice(fp1, fp2),
        kulsinski(fp1, fp2),
        yule(fp1, fp2),
        sokalmichener(fp1, fp2)
    ]
