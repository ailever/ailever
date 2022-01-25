
## Linear Algebra
### Sampling
```python
import numpy as np
import matplotlib.pyplot as plt

X = np.random.multivariate_normal(mean=[0,0], cov=[[9,0], [0, 1]], size=(100,))
display(np.cov(X.T))       # covariance matrix
display(np.corrcoef(X.T))  # correlation matrix

plt.scatter(X[:, 0], X[:, 1])
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.grid()
```

### Covariance
```python
import numpy as np
import matplotlib.pyplot as plt

X = np.random.multivariate_normal(mean=[0,0], cov=[[5,4], [0, 3]], size=(100,))
display(np.cov(X.T))       # covariance matrix
display(np.corrcoef(X.T))  # correlation matrix

plt.scatter(X[:, 0], X[:, 1])
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.grid()

eigen_values, eigen_vectors = np.linalg.eig(np.cov(X.T))
plt.quiver(
    (0, 0), 
    (0, 0), 
    eigen_vectors[:, 0],
    eigen_vectors[:, 1],
    zorder=11,
    width=0.01,
    scale=6,
    color=['r'],
)
```

### Eigen Value Problem
```python
import numpy as np
from numpy import linalg

A = np.array([[1, 2, 3],
              [4, 7, 6],
              [7, 8, 10]])

eigen_values, eigen_vectors = linalg.eig(A)
X = A@eigen_vectors
Y = eigen_values*eigen_vectors

print(np.isclose(X,Y))
print('* First eigen-vector(column vector)')
print(A@eigen_vectors[:,0])                      # left-side
print(eigen_values[0]*eigen_vectors[:,0])    # right-side

print('* Second eigen-vector(column vector)')
print(A@eigen_vectors[:,1])                      # left-side
print(eigen_values[1]*eigen_vectors[:,1])    # right-side

print('* Third eigen-vector(column vector)')
print(A@eigen_vectors[:,2])                      # left-side
print(eigen_values[2]*eigen_vectors[:,2])    # right-side
```

### Eigen Decomposition
```python
import numpy as np
from numpy import linalg

A = np.array([[1, 2, 3],
              [4, 7, 6],
              [7, 8, 10]])

eigen_values, eigen_vectors = linalg.eig(A)

Q = eigen_vectors
L = np.diag(eigen_values)
R = linalg.inv(Q)

Q@L@R
```

### SVD
```python
import numpy as np
from scipy import linalg

A = np.array([[1, 2, 3],
              [4, 7, 6],
              [7, 8, 10]])

U, s, V = linalg.svd(A)
S = np.diag(s)
U@S@V
```
