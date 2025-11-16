import numpy as np

n = int(input().strip())
data = []
for _ in range(n):
    row = list(map(float, input().strip().split()))
    data.append(row)
A = np.array(data)

vals, vecs = np.linalg.eig(A)

print(" ".join(f"{v:.3f}" for v in vals))
for row in vecs.T:
    print(" ".join(f"{x:.3f}" for x in row))
