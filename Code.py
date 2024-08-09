#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2

def compute_3d_points(depth_map, K, S):
    h, w = depth_map.shape
    points_3d = np.zeros((h, w, 3))
    for y in range(h):
        for x in range(w):
            d = depth_map[y, x]
            if d == 0:
                continue
            points_3d[y, x] = np.dot(np.linalg.inv(K), np.array([x, y, 1])) * d / S
    return points_3d

def estimate_plane_normals(points_3d):
    h, w, _ = points_3d.shape
    normals = np.zeros((h, w, 3))
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            neighborhood = points_3d[y-3:y+4, x-3:x+4].reshape(-1, 3)
            neighborhood = neighborhood[~np.all(neighborhood == 0, axis=1)]

            if len(neighborhood) < 3:
                continue

            centroid = np.mean(neighborhood, axis=0)
            centered_points = neighborhood - centroid
            _, _, V = np.linalg.svd(centered_points)
            normal = V[-1]

            if normal[-1] < 0:
                normal = -normal
            normals[y, x] = normal
    return normals

def normals_to_rgb_image(normals):
    h, w, _ = normals.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            normal = normals[y, x]
            if np.all(normal == 0):
                continue
            rgb_image[y, x] = ((normal / (2 * np.linalg.norm(normal)) + 0.5) * 255).astype(np.uint8)
    return rgb_image

# Load the depth map as a single-channel image.
depth_map = cv2.imread("depth1.png", cv2.IMREAD_GRAYSCALE)

# Define the intrinsic camera matrix K.
K = np.array([[525.0, 0, 319.5],
              [0, 525.0, 239.5],
              [0, 0, 1]])

S = 5000

# Compute the 3D points for all pixels in the image.
points_3d = compute_3d_points(depth_map, K, S)

# Estimate the plane normals for each 3D point.
normals = estimate_plane_normals(points_3d)

# Map the normals to an RGB image.
rgb_image = normals_to_rgb_image(normals)

# Save the RGB image.
cv2.imwrite("output_rgb_image.png", rgb_image)

