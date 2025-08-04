# %%

print("Importing libraries...")
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA

# %%

print("Preparing data for PCA...")
original = np.array([
  [1, 1, 1],
  [1, 2, 1],
  [2, 1, 1],
  [2, 2, 1],
  [1, 1, 2],
  [1, 2, 2],
  [2, 1, 2],
  [2, 2, 2]
])
cluster1 = original * 1 - [[1, 8, 1],
                           [1, 7, 2],
                           [1, 6, 3],
                           [1, 5, 4],
                           [1, 4, 5],
                           [1, 3, 6],
                           [1, 2, 7],
                           [1, 1, 8]]
cluster2 = original * 3
print(f"cluster1:\n{cluster1}")
print(f"cluster2:\n{cluster2}")
X = np.vstack([cluster1, cluster2])

# %%

print("Plotting original 3D data...")
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=cluster1[:, 0],
    y=cluster1[:, 1],
    z=cluster1[:, 2],
    mode='markers',
    marker=dict(size=6, color='blue'),
    name='cluster1'
))
fig.add_trace(go.Scatter3d(
    x=cluster2[:, 0],
    y=cluster2[:, 1],
    z=cluster2[:, 2],
    mode='markers',
    marker=dict(size=6, color='red'),
    name='cluster2'
))
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)
fig.show()

# %%

print("Performing PCA to reduce from 3D to 2D...")
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# %%

print("Plotting PCA result in 2D...")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=X_2d[:8, 0],
    y=X_2d[:8, 1],
    mode='markers',
    marker=dict(size=6, color='blue'),
    name='cluster1'
))
fig.add_trace(go.Scatter(
    x=X_2d[8:, 0],
    y=X_2d[8:, 1],
    mode='markers',
    marker=dict(size=6, color='red'),
    name='cluster2'
))
fig.update_layout(
    xaxis_title='PC1',
    yaxis_title='PC2',
    margin=dict(l=0, r=0, b=0, t=0)
)
fig.show()

# %%

point_idx = 1
scaling = 1.0

print("Shifting a point towards the direction of cluster2...")

original_point = X[point_idx]

cluster1_mean = cluster1.mean(axis=0)
cluster2_mean = cluster2.mean(axis=0)
direction = cluster2_mean - cluster1_mean
direction_unit = direction / np.linalg.norm(direction)

point_to_cluster2_mean = cluster2_mean - original_point
distance_to_cluster2 = np.dot(direction_unit, point_to_cluster2_mean)
adaptive_shift = distance_to_cluster2 * direction_unit

# equivalent
# proj = np.dot(direction_unit, point_to_cluster2_mean)
# total_distance = np.linalg.norm(direction)
# adaptive_shift = proj / total_distance * direction

shifted_point = original_point + scaling * adaptive_shift

original_2d = pca.transform([original_point])[0]
shifted_2d = pca.transform([shifted_point])[0]

print("Plot original and shifted point with an arrow")
fig = go.Figure()

# Plot cluster1 and cluster2 as before
fig.add_trace(go.Scatter3d(
    x=cluster1[:, 0],
    y=cluster1[:, 1],
    z=cluster1[:, 2],
    mode='markers',
    marker=dict(size=6, color='blue'),
    name='cluster1',
    text=[f"id: {i}" for i in range(len(cluster1))]  # Add id as customdata
))
fig.add_trace(go.Scatter3d(
    x=cluster2[:, 0],
    y=cluster2[:, 1],
    z=cluster2[:, 2],
    mode='markers',
    marker=dict(size=6, color='red'),
    name='cluster2',
    text=[f"id: {i+8}" for i in range(len(cluster2))]  # Add id as customdata
))

# Plot the cluster1 and cluster2 means
fig.add_trace(go.Scatter3d(
    x=[cluster1_mean[0]],
    y=[cluster1_mean[1]],
    z=[cluster1_mean[2]],
    mode='markers',
    marker=dict(size=6, color='blue', symbol='circle-open'),
    name='cluster1 mean'
))
fig.add_trace(go.Scatter3d(
   x=[cluster2_mean[0]],
   y=[cluster2_mean[1]],
   z=[cluster2_mean[2]],
   mode='markers',
   marker=dict(size=6, color='red', symbol='circle-open'),
   name='cluster2 mean',
))

# Plot the direction unit as axis
fig.add_trace(go.Scatter3d(
    x=[cluster1_mean[0] - direction_unit[0], cluster2_mean[0] + direction_unit[0]],
    y=[cluster1_mean[1] - direction_unit[1], cluster2_mean[1] + direction_unit[1]],
    z=[cluster1_mean[2] - direction_unit[2], cluster2_mean[2] + direction_unit[2]],
    mode='lines+markers',
    marker=dict(size=2, color='black'),
    line=dict(color='black', width=4, dash='dash'),
    name='direction unit axis'
))

# Plot the point_to_cluster2_mean vector
fig.add_trace(go.Scatter3d(
    x=[original_point[0], cluster2_mean[0]],
    y=[original_point[1], cluster2_mean[1]],
    z=[original_point[2], cluster2_mean[2]],
    mode='lines+markers',
    marker=dict(size=2, color='purple'),
    line=dict(color='purple', width=6, dash='dash'),
    name='point_to_cluster2_mean',
))

# Plot the shifted point
fig.add_trace(go.Scatter3d(
    x=[shifted_point[0]],
    y=[shifted_point[1]],
    z=[shifted_point[2]],
    mode='markers',
    marker=dict(size=6, color='orange', symbol='diamond'),
    name='shifted_point',
))

# Add an arrow from original_point to shifted_point
fig.add_trace(go.Scatter3d(
    x=[original_point[0], shifted_point[0]],
    y=[original_point[1], shifted_point[1]],
    z=[original_point[2], shifted_point[2]],
    mode='lines+markers',
    marker=dict(size=2, color='black'),
    line=dict(color='black', width=6, dash='dash'),
    name='shift_vector'
))

# Define and plot the dividing plane
# Create a meshgrid for the plane
xx, yy = np.meshgrid(
  np.linspace(cluster2_mean[0] - 1, cluster2_mean[0] + 1, 10), 
  np.linspace(cluster2_mean[1] - 1, cluster2_mean[1] + 1, 10)
)
# Reference: https://www.youtube.com/watch?v=2sZKZHyaQJ8
# Equation of the plane given a point
# and perpendicular normal vector:
# a(x-x0) + b(y-y0) + c(z-z0) = 0
# where (a,b,c) is the normal vector (direction_unit) 
# and (x0,y0,z0) is a point on the plane (cluster2_mean)
# We solve for z: z = z0 - (a(x-x0) + b(y-y0)) / c
a, b, c = direction_unit
x0, y0, z0 = cluster2_mean
zz = z0 - (a * (xx - x0) + b * (yy - y0)) / c

fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz,
    colorscale='Greys',
    opacity=0.5,
    showscale=False,
    name='Dividing Plane'
))

fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)
fig.show()

# %%

print("Plot PCA result in 2D with shifted point...")
fig = go.Figure()

# Plot cluster1 and cluster2 in 2D
fig.add_trace(go.Scatter(
    x=X_2d[:8, 0],
    y=X_2d[:8, 1],
    mode='markers',
    marker=dict(size=6, color='blue'),
    name='cluster1'
))
fig.add_trace(go.Scatter(
    x=X_2d[8:, 0],
    y=X_2d[8:, 1],
    mode='markers',
    marker=dict(size=6, color='red'),
    name='cluster2'
))

# Plot the cluster1 and cluster2 means in 2D
cluster1_mean_2d = pca.transform([cluster1_mean])[0]
cluster2_mean_2d = pca.transform([cluster2_mean])[0]
fig.add_trace(go.Scatter(
    x=[cluster1_mean_2d[0]],
    y=[cluster1_mean_2d[1]],
    mode='markers',
    marker=dict(size=6, color='blue', symbol='circle-open'),
    name='cluster1 mean'
))
fig.add_trace(go.Scatter(
    x=[cluster2_mean_2d[0]],
    y=[cluster2_mean_2d[1]],
    mode='markers',
    marker=dict(size=6, color='red', symbol='circle-open'),
    name='cluster2 mean'
))

# Plot the direction unit as axis in 2D
direction_unit_2d = direction_unit @ pca.components_.T
direction_unit_2d = direction_unit_2d / np.linalg.norm(direction_unit_2d)
fig.add_trace(go.Scatter(
    x=[cluster2_mean_2d[0] - direction_unit_2d[0], cluster2_mean_2d[0] + direction_unit_2d[0]],
    y=[cluster2_mean_2d[1] - direction_unit_2d[1], cluster2_mean_2d[1] + direction_unit_2d[1]],
    mode='lines+markers',
    marker=dict(size=2, color='black'),
    line=dict(color='black', width=2, dash='dash'),
    name='direction unit axis'
))

# plot the point_to_cluster2_mean vector in 2D
fig.add_trace(go.Scatter(
    x=[original_2d[0], cluster2_mean_2d[0]],
    y=[original_2d[1], cluster2_mean_2d[1]],
    mode='lines+markers',
    marker=dict(size=2, color='purple'),
    line=dict(color='purple', width=2, dash='dash'),
    name='point_to_cluster2_mean',
))

# Plot the shifted point
fig.add_trace(go.Scatter(
    x=[shifted_2d[0]],
    y=[shifted_2d[1]],
    mode='markers',
    marker=dict(size=6, color='orange', symbol='diamond'),
    name='shifted_point'
))

# Add an arrow from original_pont to shifted_point
fig.add_trace(go.Scatter(
    x=[original_2d[0], shifted_2d[0]],
    y=[original_2d[1], shifted_2d[1]],
    mode='lines+markers',
    marker=dict(size=2, color='black'),
    line=dict(color='black', width=2, dash='dash'),
    name='shift_vector'
))

fig.update_layout(
    xaxis_title='PC1',
    yaxis_title='PC2',
    margin=dict(l=0, r=0, b=0, t=0)
)
fig.show()

# %%