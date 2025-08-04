# %%

"""
Personal note:
1. What is UMAP?

   UMAP (Uniform Manifold Approximation and Projection)
   is a nonlinear dimensionality reduction technique.
   It aims to preserve both local and some global
   structure of high-dimensional data
   when projecting it to a lower-dimensional space
   (e.g., 2D for visualization). UMAP constructs
   a weighted graph representing the data's manifold
   structure, then optimizes a low-dimensional
   embedding to preserve this structure as much as
   possible. It is widely used for visualization and clusteirng,
   and is generally faster and more scalable than t-SNE.

2. Example: UMAP by Hand (3D to 2D) and Validation

   UMAP is complex and cannot be fully implemented
   in a few lines (it involves fuzzy simplicial sets,
   neighbor graphs, and stochastic optimization).
   However, you can mimic the spirit of UMAP using
   simple neighbor-preserving projection, then compare
   with the real UMAP.

   Step 1: "Manual" Neighbor-pPreserving Projection
   (for illustration)

   Suppose you have 3D points in a cluster and
   want to project to 2D, preserving nearest neighbors:
"""

import numpy as np
import matplotlib.pyplot as plt
import umap

# Generate synthetic 3D data (e.g., a noisy spiral)
np.random.seed(42)
t = np.linspace(0, 4 * np.pi, 200)
x = np.sin(t)
y = np.cos(t)
z = t + 0.1 * np.random.randn(200)
data_3d = np.vstack([x, y, z]).T

# "Manual" projection: project onto the plane defined by the first two principal axes (like PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
manual_2d = pca.fit_transform(data_3d)

print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")

# UMAP projection
umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(data_3d)

# Plot both
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].scatter(manual_2d[:, 0], manual_2d[:, 1], c=t, cmap='viridis')
axs[0].set_title('Manual (PCA-like) 2D Projection')
axs[1].scatter(umap_2d[:, 0], umap_2d[:, 1], c=t, cmap='viridis')
axs[1].set_title('UMAP 2D Projection')
plt.show()

"""
   - The manual projection (using PCA) simply projects
     onto the axes of maximum variance.
   - UMAP will try to preserve local neighbor relationships,
     possibly revealing more structure

3. Does UMAP have an "explained variance ratio" like
   PCA?

   No, UMAP does not provide an explained variance ratio.
   - PCA is a linear method and each principal component
     explains a portion of the total variance, which
     can be quantified
   - UMAP is nonlinear and focuses on preserving local
     structure (neighborhoods), not maximizing variance.
     There is no direct analog to explained variance ratio
     in UMAP.
   - If you need to quantify how much information
     is preserved, you can use trustworthiness,
     continuity, or reconstruction error,
     but these are not provided by default in UMAP.

Summary:
- UMAP is for nonlinear neighbor-preserving embedding,
  not variance maximization.
- No explained variance ratio is available for UMAP.
"""

"""
Personal note:
1. What do you mean by "It aims to preserve both local and some global structure of high-dimensional data when projecting it into a lower-dimensional space"? What do you mean by local and global structure? What is local and global structure?
   
   Local structure refers to the relationships between
   each data point and its nearest neighbors–how close
   or similar points are to each other in small
   neighborhoods.
   - Example: If two points are very close in the
     original high-dimensional space, a method
     that preserves local structure will keep them
     close in the low-dimensional embedding

   Global structure refers to the overall arangement
   and relationships between distant groups or
   clusters in the data–the "big picture" of how all
   points and clusters relate to each other
   - Example: if there are three well-separated
     clusters in high-dimensional space, a method that
     preserves global structure will keep those clusters
     separated and in roughly the same arrangement
     in the low-dimensional embedding

   In summary:
   - Local structure: Small-scale relationships (nearest
     neighbors, small clusters).
   - Global structure: Large-scale relationships
     (cluster separation, overall data geometry).

   UMAP tries to preerve botH:
   - It keeps neighbors together (local structure)
   - It also tries to maintain the relative
     positions of clusters or groups (some global
     structure), though not as strictly as local structure
"""

"""
Personal note:
1. Does PCA aims to preserve both local and some global structure of high-dimensional data when projecting it into a lower-dimensional space too?
   PCA primarily aims to preserve global structure–it
   projects data onto new axes (principal components)
   that capture the most variance in the entire dataset.
   This means PCA tries to maintain the overall shape,
   spread, and large-scale relationships of the data.

   However, PCA does not explicitly preserve local
   structure (the relationships between nearest
   neighbors). Sometimes, local structure is preserved
   as a side effect if it aligns with the directions
   of high variance, but PCA does not optimize for local
   neighborhoods

   Summary:
   - PCA: Preserves global structure (variance, overall
     geometry), but not specifically local structure.
   - UMAP: Explicitly preserves local structure 
     (neighbor relationships) and tries to maintain 
     some global structure
"""

# %%