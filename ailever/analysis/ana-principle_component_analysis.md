
## Linear Algebra
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
