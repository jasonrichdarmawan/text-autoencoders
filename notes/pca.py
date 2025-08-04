# %% [markdown]

"""
Personal note:
1. What is PCA?
   Principal Component Analysis (PCA) is a linear
   dimensionality reduction technique. It transforms
   a dataset with possibly correlated features into
   a set of linearly uncorrelated variables called
   principal components. The first principal 
   component captures the largest possible variance, 
   the second the next largest (orthogonal to the 
   first), and so on. PCA is widely used for 
   visualization, noise reduction and feature 
   extraction
"""

# %%
# Example: PCA from 3D to 2D (Manual and
# scikit-learn)

# Manual PCA (without scikit-learn)
import numpy as np

# %%

# Example 3D data (each row is a sample)
X = np.array([
  [2.5, 2.4, 0.5],
  [0.5, 0.7, 0.2],
  [2.2, 2.9, 0.7],
  [1.9, 2.2, 0.3],
  [3.1, 3.0, 0.9],
  [2.3, 2.7, 0.6],
  [2.0, 1.6, 0.4],
  [1.0, 1.1, 0.1],
  [1.5, 1.6, 0.2],
  [1.1, 0.9, 0.1]
])

# %%
"""
Personal note:
1. Why do we center the data?
   We center the data (subtract the mean of each
   feature/column) so that each feature has a mean
   of zero.
   This is important because PCA finds directions
   of maximum variance from the origin
   If the data is not centered, the first principal
   component may simply point toward the mean of
   the data, not the true direction of maximum
   variance
"""

# 1. Center the data
X_mean = np.mean(X, axis=0) # mean of each column
X_meaned = X - X_mean

print(f"X_mean:\n{X_mean}")
print(f"X_meaned:\n{X_meaned}")

# %%
"""
Personal note:
2. How is the covariance matrix computed?
   Example without `np.cov`
   The covariance matrix measures how much
   each pair of features varies together.
   For data matrix `X` (shape: samples x features),
   the covariance between feature `i` and `j` is:

    Cov(i, j) = (
      (1 / (n - 1)) 
      * \sum_{k=1}^n (X_{k,i} - \bar{X}_i) * (X_{k,j} - \bar{X}_j)
    )
"""

# Example: Compute covariance matrix manually
n_samples = X_meaned.shape[0]
cov_manual = (X_meaned.T @ X_meaned) / (n_samples - 1)

# 2. Compute covariance matrix
cov_mat = np.cov(X_meaned, rowvar=False)

print(f"Manual covariance matrix:\n{cov_manual}")
print(f"Covariance matrix:\n{cov_mat}")

# %%

"""
Personal note:
1. What are eigenvalues and eigenvectors?
   - Eigenvectors of a square matrix (A) are a special
     nonzero vectors (v) such that multiplying (A)
     by (v) only stretches or shrinks (v),
     not changing its direction: [A v = \lambda v],
     where (\lambda) is a scalar called eigenvalue
     corresponding to eigenvector (v).
   - In PCA, the eigenvectors of the covariance
     matrix point in the directions of maximum
     variance (principal components), and the
     eigenvalues tell you how much variance is in
     those directions
2. How are eigenvalues and eigenvectors computed?
   Example without `np.linalg.eigh`
   Theory
   - For a square matrix (A), eigenvalues (\lambda)
     satisfy: [ \det(A - \lambda I) = 0]. This
     is called the characteristic equation
   - For each eigenvalue, the corresponding
     eigenvector (v) solves: [ (A - \lambda I)v = 0],
"""

# 3. Comptue eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eigh(cov_mat)

print(f"eigenvalues:\n{eig_vals}")
print(f"eigenvectors:\n{eig_vecs}")

# %%

# 4. Sort eigenvectors by eigenvalues (descending)
sorted_idx = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[sorted_idx]
eig_vecs = eig_vecs[:, sorted_idx]

print("Sorted eigenvalues:\n", eig_vals)
print("Sorted eigenvectors:\n", eig_vecs)

# %%

# 5. Select top 2 eigenvectors (for 2D)
eig_vecs_2d = eig_vecs[:, :2]

"""
Personal note:
1. Why we select top 2 eigenvectors based on the
   eigen values? Why we want to use the eigenvector
   which has the largest eigen values?
   
   We select the top 2 eigenvectors based on the
   largest eigenvalues because:
   - Each eigenvector of the covariance matrix points
     in a direction in feature space
   - Each eigenvalue tells us how much variance (spread
     of the data) there is along its corresponding
     eigenvector
   
   Why use the largest eigenvalues?
   - The eigenvectors with the largest eigenvalues
     capture the most variance in the data
   - By projecting onto these directions, we retain
     as much information (variance) as possible
     in fewer dimensions
   - This is the core idea of PCA: reduce
     dimensionality while preserving the most
     important structure in the data

   Summary:
   We use the eigenvectors with the largest eigenvalues
   because they represent the directions
   where the data varies the most, which is
   what we want to keep in a lower-dimensional
   representation
"""

# %%

# 6. Project data onto new 2D space
X_pca_manual = np.dot(X_meaned, eig_vecs_2d)
print("Manual PCA result (first 2D points):\n", X_pca_manual[:3])

# %%
# scikit-learn PCA (for validation)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca_sklearn = pca.fit_transform(X)

print("sklearn PCA result (first 2D points):\n", X_pca_sklearn[:3])

# %%

explained_variance_ratio_manual = eig_vals / np.sum(eig_vals)

print("Manual PCA explained variance ratio:", explained_variance_ratio_manual)
print("PCA explained variance ratio:", pca.explained_variance_ratio_)

"""
Personal note:
3. What is PCA Explained Variance Ratio? 
   Is [0.1, 0.017] good?
   - Explained Variance Ratio: For each principal
     component, this value shows the proportion of
     the dataset's total variance captured by that
     component. For example, `[0.1, 0.017]`
     means the first component explains 10% of the
     variance, the second 1.7%
   
   Is `0.1, 0.017` good?
   - No, it's not good for visualization or
     interpretation.
     - Together, the first two components explain
       only 11.7% of the total variance.
     - This means that a 2D plot will not capture
       most of the structure in your data
     - The plot may be misleading: clusters or
       patterns you see may not reflect the
       true relationships in the high-dimensional
       space.

   Rule of thumb: For PCA plots to be meaningful,
   you typically want the first two components to
   explain a substantial portion of the variance
   (e.g., >50%). If not, be cautious in interpreting
   the plot

   Summary:
   - PCA finds the directions of maximum variance.
   - You can do PCA manually with numpy (see above)
   - Explained variance ratio tells you how much of
     the data's structure is captured
   - `[0.1, 0.017]` is low; the plot may not be
     representative of the real data structure
"""
# %%
