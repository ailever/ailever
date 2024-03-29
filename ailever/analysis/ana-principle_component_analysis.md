
## Linear Algebra
```python
import numpy as np

np.eye(n) # Unit matrix
np.diag(x) # Diagonal matrix
np.dot(a, b) # Dot product, Inner product
np.trace(x) # Trace
np.linalg.det(x) # Matrix Determinant
np.linalg.inv(x) # Inverse of a matrix
np.linalg.pinv(x) # pseudo inverse
np.linalg.eig(x) # Eigenvalue, Eigenvector
np.linalg.svd(A) # Singular Value Decomposition
np.linalg.solve(a, b) # Solve a linear matrix equation
np.linalg.lstsq(A, y, rcond=None) # Compute the Least-squares solution
```


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
print(X.T.var(axis=1, ddof=1)) # direction: x-axis, y-axis 
print(eigen_values)            # direction: eigen vector 

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

# np.cov(X.T)
# = eigen_vectors@np.diag(eigen_values)@eigen_vectors.T
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


## Principle Component Analysis
- https://www.statsmodels.org/dev/generated/statsmodels.multivariate.pca.PCA.html
- https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

### Mathematical Understanding
- https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html
```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.multivariate.pca import PCA

X = np.random.multivariate_normal(mean=[0,0], cov=[[9,0], [0, 1]], size=(100,))

pca = PCA(data=X, ncomp=2, standardize=False, demean=False, normalize=False, method='svd')
new_X = pca.factors # pca.factors ~ pca.scores ~ X@pca.eigenvecs

print(pca.eigenvals/X.shape[0])
print(pca.loadings) # pca.loadings ~ pca.eigenvecs   
print()

eigen_values, eigen_vectors = np.linalg.eig(np.cov(X.T))
print(eigen_values)
print(eigen_vectors)

plt.scatter(X[:, 0], X[:, 1])
plt.scatter(new_X[:, 0], new_X[:, 1])
```

### PCA
`statsmodels`
```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.multivariate.pca import PCA

X = np.random.multivariate_normal(mean=[0,0], cov=[[9,0], [0, 1]], size=(100,))

pca = PCA(data=X, ncomp=2, standardize=True, normalize=False, method='svd')
new_X = pca.factors

print(pca.eigenvals)
print(pca.eigenvecs)
print(pca.loadings)

plt.scatter(X[:, 0], X[:, 1])
plt.scatter(new_X[:, 0], new_X[:, 1])
```
![image](https://user-images.githubusercontent.com/56889151/151668365-c1d03823-7acb-4119-9688-5e53b3c58283.png)

`sklearn`
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

X = np.random.multivariate_normal(mean=[0,0], cov=[[9,0], [0, 1]], size=(100,))

pca = PCA(n_components=2)
new_X = pca.fit_transform(X)

print(pca.explained_variance_ratio_)
print(pca.components_)
#pca.noise_variance_

plt.scatter(X[:, 0], X[:, 1])
plt.scatter(new_X[:, 0], new_X[:, 1])
```
![image](https://user-images.githubusercontent.com/56889151/151668378-e635fa43-2750-4a32-a465-1243e5c1c6ac.png)

### Normalized PCA
`statsmodels`
```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.multivariate.pca import PCA

X = np.random.multivariate_normal(mean=[0,0], cov=[[9,0], [0, 1]], size=(100,))

pca = PCA(data=X, ncomp=2, standardize=True, normalize=True, method='svd')
new_X = pca.factors

print(pca.eigenvals)
print(pca.eigenvecs)
print(pca.loadings)

plt.scatter(X[:, 0], X[:, 1])
plt.scatter(new_X[:, 0], new_X[:, 1])
```
![image](https://user-images.githubusercontent.com/56889151/151668392-16b9e98d-54fa-4d3a-8378-c68fedefe663.png)


`sklearn`
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

X = np.random.multivariate_normal(mean=[0,0], cov=[[9,0], [0, 1]], size=(100,))

pca = Pipeline([('norm', Normalizer()), ('pca', PCA(n_components=2))])
new_X = pca.fit_transform(X)

print(pca.named_steps['pca'].explained_variance_ratio_)
print(pca.named_steps['pca'].components_)
#pca.named_steps['pca'].noise_variance_

plt.scatter(X[:, 0], X[:, 1])
plt.scatter(new_X[:, 0], new_X[:, 1])
```
![image](https://user-images.githubusercontent.com/56889151/151668408-bb845021-7f78-4db5-9957-16ea1ad400ea.png)



## Covariance Estimation
- https://scikit-learn.org/stable/modules/covariance.html
```python
```


