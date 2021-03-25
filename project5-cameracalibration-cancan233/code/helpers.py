import cv2
import numpy as np
from skimage import img_as_float32
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random


def get_markers(markers_path):
    """
    Returns a dictionary mapping a marker ID to a 4x3 array
    containing the 3d points for each of the 4 corners of the
    marker in our scanning setup
    """
    markers = {}
    with open(markers_path) as f:
        first_dim = 0
        second_dim = 0
        for i, line in enumerate(f.readlines()):
            if i == 0:
                first_dim, second_dim = [float(x) for x in line.split()]
            else:
                info = [float(x) for x in line.split()]
                markers[i] = [
                    [info[0], info[1], info[2]],
                    [info[0] + first_dim * info[3], info[1] + first_dim * info[4], info[2] + first_dim * info[5]],
                    [info[0] + first_dim * info[3] + second_dim * info[6], info[1] + first_dim * info[4] + second_dim * info[7], info[2] + first_dim * info[5] + second_dim * info[8]],
                    [info[0] + second_dim * info[6], info[1] + second_dim * info[7], info[2] + second_dim * info[8]],
                ]
    return markers


def get_matches(image1, image2, num_keypoints=5000):
    """
    # Wraps OpenCV's ORB function and feature matcher
    # returns two N x 2 numpy arrays, 2d points in image1 and image2
    # that are proposed matches
    """
    # find the keypoints and descriptors with ORB
    orb = cv2.ORB_create(nfeatures=num_keypoints)
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]
    list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]
    matches_kp1 = np.asarray(list_kp1)
    matches_kp2 = np.asarray(list_kp2)

    # Remove duplicate matches
    combine_reduce = np.unique(np.concatenate((matches_kp1, matches_kp2),
                                             axis=1),
                              axis=0)
    points1 = combine_reduce[:, 0:2]
    points2 = combine_reduce[:, -2:]

    return points1, points2


def show_matches(image1, image2, points1, points2):
    """
    Shows matches from image1 to image2, represented by Nx2 arrays
    points1 and points2
    """
    image1 = img_as_float32(image1)
    image2 = img_as_float32(image2)

    fig = plt.figure()
    plt.axis('off')

    matches_image = np.hstack([image1, image2])
    plt.imshow(matches_image)

    shift = image1.shape[1]
    for i in range(0, points1.shape[0]):

        random_color = lambda: random.randint(0, 255)
        cur_color = ('#%02X%02X%02X' % (random_color(), random_color(), random_color()))

        x1 = points1[i, 1]
        y1 = points1[i, 0]
        x2 = points2[i, 1]
        y2 = points2[i, 0]

        x = np.array([x1, x2])
        y = np.array([y1, y2 + shift])
        plt.plot(y, x, c=cur_color, linewidth=0.5)

    plt.show()


def reproject_points(M, points):
    """
    Use projection matrix to project Nx3 array of 3d points into Nx2
    array of image points
    """

    reshaped_points = np.concatenate(
        (points, np.ones((points.shape[0], 1))), axis=1)
    projected_points = np.matmul(M, np.transpose(reshaped_points))
    projected_points = np.transpose(projected_points)
    u = np.divide(projected_points[:, 0], projected_points[:, 2])
    v = np.divide(projected_points[:, 1], projected_points[:, 2])
    projected_points = np.transpose(np.vstack([u, v]))

    return projected_points


def show_reprojections(images, Ms, markers):
    """
    Show reprojected markers in each image
    """
    points3d = []

    for marker_id in markers:
        points3d += markers[marker_id]
    points3d = np.array(points3d)

    fig, axs = plt.subplots(1, len(images), figsize=(15, 6))
    plt.axis('off')

    for i in range(len(images)):
        points2d = reproject_points(Ms[i], points3d)
        axs[i].imshow(images[i])
        axs[i].scatter(points2d[:, 0], points2d[:, 1])
    plt.show()


def show_point_cloud(points3d, colors):
    """
    Show 3D points with their corresponding colors
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=points3d[:, 0],
        y=points3d[:, 1],
        z=points3d[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            opacity=1
        )
    )])

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
