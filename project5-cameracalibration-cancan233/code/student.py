import numpy as np
import cv2
from random import sample


def calculate_projection_matrix(image, markers):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points. See the handout, Q5
    of the written questions, or the lecture slides for how to set up these
    equations.

    Don't forget to set M_34 = 1 in this system to fix the scale.

    :param image: a single image in our camera system
    :param markers: dictionary of markerID to 4x3 array containing 3D points
    :return: M, the camera projection matrix which maps 3D world coordinates
    of provided aruco markers to image coordinates
    """
    ######################
    # Do not change this #
    ######################

    # Markers is a dictionary mapping a marker ID to a 4x3 array
    # containing the 3d points for each of the 4 corners of the
    # marker in our scanning setup
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        image, dictionary, parameters=parameters)
    markerIds = [m[0] for m in markerIds]
    markerCorners = [m[0] for m in markerCorners]

    points2d = []
    points3d = []

    for markerId, marker in zip(markerIds, markerCorners):
        if markerId in markers:
            for j, corner in enumerate(marker):
                points2d.append(corner)
                points3d.append(markers[markerId][j])

    points2d = np.array(points2d)
    points3d = np.array(points3d)

    ########################
    # TODO: Your code here #
    ########################
    # This M matrix came from a call to rand(3,4). It leads to a high residual.
    print('Randomly setting matrix entries as a placeholder')
    M = np.array([[0.1768, 0.7018, 0.7948, 0.4613],
                  [0.6750, 0.3152, 0.1136, 0.0480],
                  [0.1020, 0.1725, 0.7244, 0.9932]])

    return M


def normalize_coordinates(points):
    """
    ============================ EXTRA CREDIT ============================
    Normalize the given Points before computing the fundamental matrix. You
    should perform the normalization to make the mean of the points 0
    and the average magnitude 1.0.

    The transformation matrix T is the product of the scale and offset matrices

    Offset Matrix
    Find c_u and c_v and create a matrix of the form in the handout for T_offset

    Scale Matrix
    Subtract the means of the u and v coordinates, then take the reciprocal of
    their standard deviation i.e. 1 / np.std([...]). Then construct the scale
    matrix in the form provided in the handout for T_scale

    :param points: set of [n x 2] 2D points
    :return: a tuple of (normalized_points, T) where T is the [3 x 3] transformation
    matrix
    """
    ########################
    # TODO: Your code here #
    ########################
    # This is a placeholder with the identity matrix for T replace with the
    # real transformation matrix for this set of points
    T = np.eye(3)

    return points, T


def estimate_fundamental_matrix(points1, points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Try to implement this function as efficiently as possible. It will be
    called repeatedly for part IV of the project

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
    """
    ########################
    # TODO: Your code here #
    ########################

    # This is an intentionally incorrect Fundamental matrix placeholder
    F_matrix = np.array([[0, 0, -.0004], [0, 0, .0032], [0, -0.0044, .1034]])

    return F_matrix


def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Run RANSAC for num_iters.

    matches1 and matches2 are the [N x 2] coordinates of the possibly
    matching points from two pictures. Each row is a correspondence
     (e.g. row 42 of matches1 is a point that corresponds to row 42 of matches2)

    best_Fmatrix is the [3 x 3] fundamental matrix, inliers1 and inliers2 are
    the [M x 2] corresponding points (some subset of matches1 and matches2) that
    are inliners with respect to best_Fmatrix

    For this section, use RANSAC to find the best fundamental matrix by randomly
    sampling interest points. You would reuse estimate_fundamental_matrix from
    Part 2 of this assignment.

    If you are trying to produce an uncluttered visualization of epipolar lines,
    you may want to return no more than 30 points for either image.

    :return: best_Fmatrix, inliers1, inliers2
    """
    ########################
    # TODO: Your code here #
    ########################

    # Your RANSAC loop should contain a call to 'estimate_fundamental_matrix()'
    # that you wrote for part II.

    best_Fmatrix = estimate_fundamental_matrix(matches1[0:9, :], matches2[0:9, :])
    inliers_a = matches1[0:29, :]
    inliers_b = matches2[0:29, :]

    return best_Fmatrix, inliers_a, inliers_b


def matches_to_3d(points1, points2, M1, M2):
    """
    Given two sets of points and two projection matrices, you will need to solve
    for the ground-truth 3D points using np.linalg.lstsq(). For a brief reminder
    of how to do this, please refer to Question 5 from the written questions for
    this project.


    :param points1: [N x 2] points from image1
    :param points2: [N x 2] points from image2
    :param M1: [3 x 4] projection matrix of image2
    :param M2: [3 x 4] projection matrix of image2
    :return: [N x 3] list of solved ground truth 3D points for each pair of 2D
    points from points1 and points2
    """
    points3d = []
    ########################
    # TODO: Your code here #
    ########################

    return points3d
