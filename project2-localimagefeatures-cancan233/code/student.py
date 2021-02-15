import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
import math


def get_interest_points(image, feature_width):
    """
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!

    # STEP 1: Calculate the gradient (partial derivatives on two directions).
    # STEP 2: Apply Gaussian filter with appropriate sigma.
    # STEP 3: Calculate Harris cornerness score for all pixels.
    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)

    # BONUS: There are some ways to improve:
    # 1. Making feature detection multi-scaled.
    # 2. Use adaptive non-maximum suppression.

    # filter image with narrow guassian fliter for better edge detection
    sigma = 1.65

    filtered_image = filters.gaussian(image, sigma=1.0)
    Iy = filters.sobel_h(filtered_image)
    Ix = filters.sobel_v(filtered_image)
    Ixx = filters.gaussian(Ix * Ix, sigma=sigma)
    Ixy = filters.gaussian(Ix * Iy, sigma=sigma)
    Iyy = filters.gaussian(Iy * Iy, sigma=sigma)

    # R = det(M) - k(trace(M))^2
    # det(M) = AB - C^2, where A = filter(Ixx), B = filter(Iyy), C = filter(Ixy)
    # trace(M) = A + B
    alpha = 0.04
    R = Ixx * Iyy - Ixy ** 2 - alpha * (Ixx + Iyy) ** 2
    # R = (R - np.min(R)) / (np.max(R) - np.min(R))
    # mask = R < np.mean(R)
    # R[mask] = 0
    interested_points = feature.peak_local_max(
        R,
        min_distance=feature_width // 2,
        threshold_abs=1e-6,
        exclude_border=True,
        num_peaks=2000,
    )
    # print(len(interested_points))
    # interested_points = anms(interested_points, R)
    # print(len(interested_points))
    return interested_points[:, 1], interested_points[:, 0]


def anms(points, harris_response, top=500):
    l, x, y = [], 0, 0
    threshold = np.mean(harris_response)
    while x < len(points):
        minpoint = float("inf")
        xi, yi = points[x][0], points[x][1]
        while y < len(points):
            xj, yj = points[y][0], points[y][1]
            if (
                xi != xj
                and yi != yj
                and harris_response[points[x][0], points[x][1]] > threshold
                and harris_response[points[y][0], points[y][1]] > threshold
                and harris_response[points[x][0], points[x][1]]
                < harris_response[points[y][0], points[y][1]] * 0.9
            ):
                dist = math.sqrt((xj - xi) ** 2 + (yj - yi) ** 2)
                if dist < minpoint:
                    minpoint = dist
            y += 1
        l.append([xi, yi, minpoint])
        x += 1
        y = 0
    l.sort(key=lambda x: x[2], reverse=True)
    return np.array(l[:top])


def get_features(image, x, y, feature_width):
    """
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels.
    # STEP 2: Decompose the graident vectors to magnitude and direction.
    # STEP 3: For each feature point, calculate the local histogram based on related 4x4 grid cells.
    #         Each cell is a square with feature_width / 4 pixels length of side.
    #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
    #         based on the direction (angle) of the gradient vectors.
    # STEP 4: Now for each cell, we have a 8-dimensional vector. Appending the vectors in the 4x4 cells,
    #         we have a 128-dimensional descriptor.
    # STEP 5: Don't forget to normalize your descriptor.

    # BONUS: There are some ways to improve:
    # 1. Use multi-scaled descriptor.
    # 2. Borrow ideas from GLOH or other type of descriptors.

    # This is a placeholder - replace this with your features!

    if feature_width % 4 != 0:
        raise ValueError("feature_width must be a multiple of 4.")

    x = np.round(x).astype(int).flatten()
    y = np.round(y).astype(int).flatten()
    num_bins = 8
    bins = np.arange(0, 2 * np.pi, 2 * np.pi / num_bins)
    filtered_image = filters.gaussian(image, sigma=1.0)
    Iy = filters.sobel_h(filtered_image)
    Ix = filters.sobel_v(filtered_image)
    gradients = np.stack((Iy, Ix), axis=0)
    magnitudes = np.linalg.norm(gradients, axis=0)
    orientations = np.arctan2(gradients[0], gradients[1])
    offset = feature_width // 2
    features = []

    # construct SIFT-like descriptor
    for ix, iy in zip(x, y):
        window_magnitudes = magnitudes[
            iy - offset + 1 : iy + offset + 1, ix - offset + 1 : ix + offset + 1
        ]
        if window_magnitudes.shape != (feature_width, feature_width):
            continue
        window_orientations = orientations[
            iy - offset + 1 : iy + offset + 1, ix - offset + 1 : ix + offset + 1
        ]

        split_window_magnitudes = np.array(
            np.split(
                np.array(np.split(window_magnitudes, 4, axis=1)).reshape(4, -1),
                4,
                axis=1,
            )
        ).reshape(-1, feature_width)

        split_window_orientations = np.array(
            np.split(
                np.array(np.split(window_orientations, 4, axis=1)).reshape(4, -1),
                4,
                axis=1,
            )
        ).reshape(-1, feature_width)
        feature = np.zeros(int(feature_width * feature_width * num_bins / 16))

        for subwindow_i in range(split_window_magnitudes.shape[0]):
            inds = np.digitize(split_window_orientations[subwindow_i], bins)
            for inds_i in range(num_bins):
                mask = np.array(inds == inds_i)
                feature[subwindow_i * num_bins + inds_i] = np.sum(
                    split_window_magnitudes[subwindow_i].flatten()[mask]
                )

        feature = feature ** 0.6
        feature_norm = feature / np.linalg.norm(feature)
        np.putmask(feature_norm, feature_norm < 0.001, 0)
        feature_norm = feature_norm ** 0.7
        feature_norm_2 = feature_norm / np.linalg.norm(feature_norm)
        features.append(feature_norm_2)

    # construct GLOH descriptor

    return np.array(features)


def match_features(im1_features, im2_features):
    """
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with your matches and confidences!

    # STEP 1: Calculate the distances between each pairs of features between im1_features and im2_features.
    #         HINT: https://docs.google.com/document/d/1SlzMaiS4rq6M8ySDXZTgUH_tyVV2rBQQzb_c1PQZfKI/edit
    # STEP 2: Sort and find closest features for each feature, then performs NNDR test.

    # BONUS: Using PCA might help the speed (but maybe not the accuracy).

    # im1_features = PCA(im1_features, 128)
    # im2_features = PCA(im2_features, 128)

    # D = sqrt(F1^2 + F2^2 - 2 * F1 * F2)
    B = 2 * np.matmul(im1_features, im2_features.transpose())
    F1_square = np.sum(np.square(im1_features), axis=1, keepdims=True)
    F2_square = np.sum(np.square(im2_features), axis=1, keepdims=True).transpose()

    A = F1_square + F2_square
    D = np.sqrt(A - B)

    # perform NNDR test
    idx1 = np.sort(D, axis=1)[:, :2]
    nddr1 = idx1[:, 0] / idx1[:, 1]
    confidences1 = 1 - nddr1

    closest_ind1 = np.argmin(D, axis=1)
    matches1 = np.stack((np.arange(D.shape[0]), closest_ind1), axis=1)

    idx2 = np.sort(D.T, axis=1)[:, :2]
    nddr2 = idx2[:, 0] / idx2[:, 1]
    confidences2 = 1 - nddr2

    closest_ind2 = np.argmin(D.T, axis=1)
    matches2 = np.stack((closest_ind2, np.arange(D.T.shape[0])), axis=1)

    # cross check match pair
    idx_pair = matches1[matches2[:, 0]][:, 1] == matches2[:, 1]
    matches = matches2[idx_pair]
    confidences = (confidences1[matches2[idx_pair, 0]] + confidences2[idx_pair]) / 2

    return np.array(matches), np.array(confidences)


def PCA(features, m):
    M = np.mean(features.T, axis=1)
    C = features - M
    cov_matrix = np.cov(C.T)
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
    i = np.argsort(eigenvals)[::-1]
    eigenvecs = eigenvecs[:, i]
    eigenvals = eigenvals[i]
    pca_features = np.dot(C, eigenvecs[:m])
    return pca_features