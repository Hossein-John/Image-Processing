import numpy as np

def eigen_finder(matrix):

    n, m = matrix.shape
    if n != m:
        return []
    
    eigen_values, eigen_vectors = np.linalg.eig(matrix)
    return eigen_values.tolist()

def frobenious_norm_finder(matrix):
    frobenius_norm = float(np.linalg.norm(matrix, 'fro'))
    return frobenius_norm

def infinity_norm_finder(matrix):
    infinity_norm = float(np.linalg.norm(matrix, np.inf))
    return infinity_norm

def min_max_normalizer(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix.tolist()
