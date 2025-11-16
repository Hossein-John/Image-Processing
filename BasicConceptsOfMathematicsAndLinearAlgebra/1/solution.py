# solution.py
import numpy as np

def addition(A: np.ndarray, B: np.ndarray):
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        return "Cannot be done"
    if A.shape != B.shape:
        return "Cannot be done"
    return A + B

def subtraction(A: np.ndarray, B: np.ndarray):
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        return "Cannot be done"
    if A.shape != B.shape:
        return "Cannot be done"
    return A - B

def multiply(A: np.ndarray, B: np.ndarray):
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        return "Cannot be done"
    if A.ndim != 2 or B.ndim != 2:
        return "Cannot be done"
    if A.shape[1] != B.shape[0]:
        return "Cannot be done"
    return A @ B

def convolution(X: np.ndarray, K: np.ndarray):
    if not isinstance(X, np.ndarray) or not isinstance(K, np.ndarray):
        return "Cannot be done"
    if X.ndim != 2 or K.ndim != 2:
        return "Cannot be done"

    H, W = X.shape
    kh, kw = K.shape
    if kh > H or kw > W:
        return "Cannot be done"

    out_h, out_w = H - kh + 1, W - kw + 1
    out = np.zeros((out_h, out_w), dtype=np.result_type(X.dtype, K.dtype))

    for i in range(out_h):
        for j in range(out_w):
            window = X[i:i+kh, j:j+kw]
            out[i, j] = np.sum(window * K)

    return out
