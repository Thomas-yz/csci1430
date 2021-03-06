import re
import numpy as np
import matplotlib
import time
from numpy.core.defchararray import mod
from numpy.lib.function_base import average
from numpy.ma.core import power
from scipy.ndimage.measurements import label
from scipy import ndimage as ndi
from scipy.stats.stats import KendalltauResult

from skimage import feature
from helpers import progressbar
from skimage.io import imread
from skimage.color import rgb2grey
from skimage.feature import hog, daisy
from skimage.transform import resize, pyramid_gaussian
from skimage.filters import gabor_kernel, gaussian, sobel_h, sobel_v

from scipy.spatial.distance import cdist
from scipy.stats import mode

from sklearn import cluster
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectPercentile, chi2
from skelarn.mixture import GaussianMixture as GMM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# from sklearn.mixture import GaussianMixture

from joblib import Parallel, delayed


def get_tiny_images(image_paths):
    """
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    Inputs:
        image_paths: a 1-D Python list of strings. Each string is a complete
                     path to an image on the filesystem.
    Outputs:
        An n x d numpy array where n is the number of images and d is the
        length of the tiny image representation vector. e.g. if the images
        are resized to 16x16, then d is 16 * 16 = 256.

    To build a tiny image feature, resize the original image to a very small
    square resolution (e.g. 16x16). You can either resize the images to square
    while ignoring their aspect ratio, or you can crop the images into squares
    first and then resize evenly. Normalizing these tiny images will increase
    performance modestly.

    As you may recall from class, naively downsizing an image can cause
    aliasing artifacts that may throw off your comparisons. See the docs for
    skimage.transform.resize for details:
    http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize

    Suggested functions: skimage.transform.resize, skimage.color.rgb2grey,
                         skimage.io.imread, np.reshape
    """

    # TODO: Implement this function!
    images = []
    for image_path in image_paths:
        image = imread(image_path, as_gray=True)
        images.append(image)
    output = Parallel(n_jobs=-1)(delayed(tiny_feature)(i) for i in images)
    return np.array(output)


def build_vocabulary(image_paths, vocab_size):
    """
    This function should sample HOG descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Inputs:
        image_paths: a Python list of image path strings
         vocab_size: an integer indicating the number of words desired for the
                     bag of words vocab set

    Outputs:
        a vocab_size x (z*z*9) (see below) array which contains the cluster
        centers that result from the K Means clustering.

    You'll need to generate HOG features using the skimage.feature.hog() function.
    The documentation is available here:
    http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog

    However, the documentation is a bit confusing, so we will highlight some
    important arguments to consider:
        cells_per_block: The hog function breaks the image into evenly-sized
            blocks, which are further broken down into cells, each made of
            pixels_per_cell pixels (see below). Setting this parameter tells the
            function how many cells to include in each block. This is a tuple of
            width and height. Your SIFT implementation, which had a total of
            16 cells, was equivalent to setting this argument to (4,4).
        pixels_per_cell: This controls the width and height of each cell
            (in pixels). Like cells_per_block, it is a tuple. In your SIFT
            implementation, each cell was 4 pixels by 4 pixels, so (4,4).
        feature_vector: This argument is a boolean which tells the function
            what shape it should use for the return array. When set to True,
            it returns one long array. We recommend setting it to True and
            reshaping the result rather than working with the default value,
            as it is very confusing.

    It is up to you to choose your cells per block and pixels per cell. Choose
    values that generate reasonably-sized feature vectors and produce good
    classification results. For each cell, HOG produces a histogram (feature
    vector) of length 9. We want one feature vector per block. To do this we
    can append the histograms for each cell together. Let's say you set
    cells_per_block = (z,z). This means that the length of your feature vector
    for the block will be z*z*9.

    With feature_vector=True, hog() will return one long np array containing every
    cell histogram concatenated end to end. We want to break this up into a
    list of (z*z*9) block feature vectors. We can do this using a really nifty numpy
    function. When using np.reshape, you can set the length of one dimension to
    -1, which tells numpy to make this dimension as big as it needs to be to
    accomodate to reshape all of the data based on the other dimensions. So if
    we want to break our long np array (long_boi) into rows of z*z*9 feature
    vectors we can use small_bois = long_boi.reshape(-1, z*z*9).

    The number of feature vectors that come from this reshape is dependent on
    the size of the image you give to hog(). It will fit as many blocks as it
    can on the image. You can choose to resize (or crop) each image to a consistent size
    (therefore creating the same number of feature vectors per image), or you
    can find feature vectors in the original sized image.

    ONE MORE THING
    If we returned all the features we found as our vocabulary, we would have an
    absolutely massive vocabulary. That would make matching inefficient AND
    inaccurate! So we use K Means clustering to find a much smaller (vocab_size)
    number of representative points. We recommend using sklearn.cluster.KMeans
    (or sklearn.cluster.MiniBatchKMeans if KMeans takes to long for you) to do this.
    Note that this can take a VERY LONG TIME to complete (upwards of ten minutes
    for large numbers of features and large max_iter), so set the max_iter argument
    to something low (we used 100) and be patient. You may also find success setting
    the "tol" argument (see documentation for details)
    """

    # TODO: Implement this function!
    # vocab_size = 10000
    num_imgs = len(image_paths)
    images = []
    for i in range(num_imgs):
        image = imread(image_paths[i], as_gray=True)
        images.append(image)

    features = []
    descriptors = [
        "hog",
        "gaussian_pyramid",
        # "gist",
        # "daisy",
    ]
    ################### baseline implementation ################
    if "hog" in descriptors:
        output = Parallel(n_jobs=-1)(delayed(hog_feature)(i) for i in images)
        output = np.concatenate(np.array(output), axis=0)
        features.append(output)

    ################## pyramid gaussian #####################
    if "gaussian_pyramid" in descriptors:
        output = Parallel(n_jobs=-1)(
            delayed(gaussian_pyramid_feature)(i) for i in images
        )
        output = np.concatenate(np.array(output), axis=0)
        features.append(output)

    #################### gist descriptor #####################
    if "gist" in descriptors:
        output = Parallel(n_jobs=-1)(delayed(gist_feature)(i) for i in images)
        output = np.concatenate(np.array(output), axis=0)
        features.append(output)

    if "daisy" in descriptors:
        output = Parallel(n_jobs=-1)(delayed(daisy_feature)(i) for i in images)
        output = np.concatenate(np.array(output), axis=0)
        features.append(output)

    if "sift" in descriptors:
        output = Parallel(n_jobs=-1)(delayed(daisy_feature)(i) for i in images)
        output = np.concatenate(np.array(output), axis=0)
        features.append(output)

    features = np.concatenate(np.array(features), axis=0)
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=vocab_size,
        init_size=3 * max(vocab_size, 100),
        max_iter=100,
    )
    kmeans.fit(features)
    vocab = kmeans.cluster_centers_
    return vocab


def get_bags_of_words(image_paths):
    """
    This function should take in a list of image paths and calculate a bag of
    words histogram for each image, then return those histograms in an array.

    Inputs:
        image_paths: A Python list of strings, where each string is a complete
                     path to one image on the disk.

    Outputs:
        An nxd numpy matrix, where n is the number of images in image_paths and
        d is size of the histogram built for each image.

    Use the same hog function to extract feature vectors as before (see
    build_vocabulary). It is important that you use the same hog settings for
    both build_vocabulary and get_bags_of_words! Otherwise, you will end up
    with different feature representations between your vocab and your test
    images, and you won't be able to match anything at all!

    After getting the feature vectors for an image, you will build up a
    histogram that represents what words are contained within the image.
    For each feature, find the closest vocab word, then add 1 to the histogram
    at the index of that word. For example, if the closest vector in the vocab
    is the 103rd word, then you should add 1 to the 103rd histogram bin. Your
    histogram should have as many bins as there are vocabulary words.

    Suggested functions: scipy.spatial.distance.cdist, np.argsort,
                         np.linalg.norm, skimage.feature.hog
    """

    vocab = np.load("vocab.npy")
    print("Loaded vocab from file.")

    # TODO: Implement this function!
    num_imgs = len(image_paths)

    images = []
    for i in range(num_imgs):
        image = imread(image_paths[i], as_gray=True)
        images.append(image)

    ######################## baseline implementation ################
    # task_labels: ith image, vocab, optional:"hog", "gist", "gaussian_pyramid"
    output = Parallel(n_jobs=-1)(
        delayed(generate_feats)(
            i,
            vocab,
            "hog",
            "gaussian_pyramid",
            # "gist",
            # "daisy",
        )
        for i in images
    )
    return np.array(output)


def svm_classify(train_image_feats, train_labels, test_image_feats):
    """
    This function will predict a category for every test image by training
    15 many-versus-one linear SVM classifiers on the training data, then
    using those learned classifiers on the testing data.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy array of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

    We suggest you look at the sklearn.svm module, including the LinearSVC
    class. With the right arguments, you can get a 15-class SVM as described
    above in just one call! Be sure to read the documentation carefully.
    """

    # TODO: Implement this function!
    # svm_model = LinearSVC(tol=1e-8)
    svm_model = SVC(
        kernel="poly",
        # C=0.5,
        degree=5,
        gamma=0.5,
        coef0=1.0,
        shrinking=True,
        probability=True,
        tol=1e-8,
        class_weight="balanced",
        break_ties=True,
    )
    # svm_model = SVC(
    #     kernel="sigmoid",
    #     gamma=0.5,
    #     coef0=1.0,
    #     tol=1e-8,
    #     class_weight="balanced",
    #     break_ties=True,
    # )
    # svm_model = SVC(
    #     kernel="rbf",
    #     gamma=0.5,
    #     tol=1e-8,
    # )
    # svm_model = Pipeline(
    #     [
    #         ("anova", SelectPercentile(chi2, percentile=50)),
    #         ("scalar", StandardScaler()),
    #         (
    #             "svc",
    #             SVC(
    #                 kernel="poly",
    #                 degree=5,
    #                 gamma=0.5,
    #                 coef0=1.0,
    #                 shrinking=True,
    #                 probability=True,
    #                 tol=1e-8,
    #                 class_weight="balanced",
    #                 break_ties=True,
    #             ),
    #         ),
    #     ]
    # )
    svm_model.set_params()
    svm_model.fit(train_image_feats, train_labels)
    labels = svm_model.predict(test_image_feats)
    return labels


def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    """
    This function will predict the category for every test image by finding
    the training image with most similar features. You will complete the given
    partial implementation of k-nearest-neighbors such that for any arbitrary
    k, your algorithm finds the closest k neighbors and then votes among them
    to find the most common category and returns that as its prediction.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy list of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

    The simplest implementation of k-nearest-neighbors gives an even vote to
    all k neighbors found - that is, each neighbor in category A counts as one
    vote for category A, and the result returned is equivalent to finding the
    mode of the categories of the k nearest neighbors. A more advanced version
    uses weighted votes where closer matches matter more strongly than far ones.
    This is not required, but may increase performance.

    Be aware that increasing k does not always improve performance - even
    values of k may require tie-breaking which could cause the classifier to
    arbitrarily pick the wrong class in the case of an even split in votes.
    Additionally, past a certain threshold the classifier is considering so
    many neighbors that it may expand beyond the local area of logical matches
    and get so many garbage votes from a different category that it mislabels
    the data. Play around with a few values and see what changes.

    Useful functions:
        scipy.spatial.distance.cdist, np.argsort, scipy.stats.mode
    """

    k = 1

    # Gets the distance between each test image feature and each train image feature
    # e.g., cdist
    distances = cdist(test_image_feats, train_image_feats, "euclidean")

    # TODO:
    # 1) Find the k closest features to each test image feature in euclidean space
    # 2) Determine the labels of those k features
    # 3) Pick the most common label from the k
    # 4) Store that label in a list

    k = 15
    #################### baseline vote ##############
    # nearest_neighbor_idx = np.argsort(distances, axis=1)[:, :k]
    # nearest_neighbor_labels = np.array(train_labels)[nearest_neighbor_idx]
    # labels = mode(nearest_neighbor_labels, axis=1)[0]

    ############## weighted votes #############
    m = test_image_feats.shape[0]
    distances_labels = np.tile(train_labels, (m, 1))
    k_nearest_neighbor_idx = np.argsort(distances, axis=1)[:, :k]
    k_nearest_distances = np.take_along_axis(distances, k_nearest_neighbor_idx, axis=1)
    k_labels = np.take_along_axis(distances_labels, k_nearest_neighbor_idx, axis=1)
    weights = 1 / (k_nearest_distances + 1e-8)
    weights /= np.sum(weights, axis=1)[:, np.newaxis]
    np.putmask(weights, weights < 0.1 / k, 0)

    unique_labels = np.array(list(set(train_labels)))
    votes = np.zeros((m, len(unique_labels)))
    for i in range(len(unique_labels)):
        votes[:, i] = np.sum(np.where(k_labels == unique_labels[i], weights, 0), axis=1)
    labels_idx = np.argmax(votes, axis=1)
    labels = unique_labels[labels_idx]

    return labels


def tiny_feature(image, *args, **kwargs):
    image = resize(image, (16, 16))
    return image.flatten()


def hog_feature(image, *args, **kwargs):
    pixels_per_cell_dim = 8
    cells_per_block_dim = 2
    orientation_bin = 9
    feature = hog(
        image,
        orientations=orientation_bin,
        pixels_per_cell=(pixels_per_cell_dim, pixels_per_cell_dim),
        cells_per_block=(cells_per_block_dim, cells_per_block_dim),
        feature_vector=True,
    ).reshape(-1, cells_per_block_dim * cells_per_block_dim * orientation_bin)
    return feature


def gaussian_pyramid_feature(image, *args, **kwargs):
    pixels_per_cell_dim = 8
    cells_per_block_dim = 2
    for (_, resized) in enumerate(pyramid_gaussian(image, downscale=2, max_layer=3)):
        resized_image = resize(resized, image.shape)
        feature = hog(
            resized_image,
            orientations=9,
            pixels_per_cell=(pixels_per_cell_dim, pixels_per_cell_dim),
            cells_per_block=(cells_per_block_dim, cells_per_block_dim),
            feature_vector=True,
        ).reshape(-1, cells_per_block_dim * cells_per_block_dim * 9)
    return feature


def gist_feature(image):
    """
    Given an input image, a GIST descriptor is computed by
    1.Convolve the image with 32 Gabor filters at 4 scales, 8 orientations, producing 32 feature maps of the same size of the input image.
    2.Divide each feature map into 16 regions (by a 4x4 grid), and then average the feature values within each region.
    3.Concatenate the 16 averaged values of all 32 feature maps, resulting in a 16x32=512 GIST descriptor.
    Intuitively, GIST summarizes the gradient information (scales and orientations) for different parts of an image, which provides a rough description (the gist) of the scene.
    """

    # square image
    r, c = image.shape
    dim = (min(r, c) // 4) * 4
    image = resize(image, (dim, dim))

    kernels = []
    for theta in range(8):
        theta = theta / 8.0 * np.pi
        for frequency in (0.1, 0.2, 0.3, 0.4):
            kernels.append(np.real(gabor_kernel(frequency=frequency, theta=theta)))

    features = []
    for kernel in kernels:
        filtered_image = np.sqrt(
            ndi.convolve(image, np.real(kernel), mode="wrap") ** 2
            + ndi.convolve(image, np.imag(kernel), mode="wrap") ** 2
        )
        patch_filtered_image = np.array(
            np.hsplit(np.array(np.hsplit(filtered_image, 4)), 4)
        ).reshape(16, -1)
        feature = np.mean(patch_filtered_image, axis=1)
        features.extend(feature)
    # only take 504 as 512 cannot be divided by 36
    features = np.array(features)[: -(512 % 36)]
    return features.reshape((-1, 36))


def daisy_feature(image):
    feature = daisy(image).reshape(-1, 200)
    return feature


def generate_feats(image, vocab, *args, **kwargs):
    feature = []
    if "hog" in args:
        feature.append(hog_feature(image))

    if "gist" in args:
        feature.append(gist_feature(image))

    if "gaussian_pyramid" in args:
        feature.append(gaussian_pyramid_feature(image))

    if "daisy" in args:
        feature.append(daisy_feature(image))

    feature = np.concatenate(np.array(feature), axis=0)
    distances = cdist(feature, vocab, "euclidean")

    # vocab_idx = np.argmin(distances, axis=1).flatten()
    # feats = np.bincount(vocab_idx, minlength=len(vocab))
    # feats = feats / np.linalg.norm(feats)

    #################### Soft asssignment ###############################
    # Use "soft assignment" to assign visual words to histogram bins. Each
    # visual word will cast a distance-weighted vote to multiple bins.

    sigma = 0.05
    distances = np.exp(-(distances ** 2) / (sigma ** 2))
    distances = distances / np.sum(distances, axis=1)[:, np.newaxis]
    feats = np.sum(distances, axis=0)

    feats = feats / np.linalg.norm(feats)
    return feats


def fisher_vector(xx):
    gmm = GMM(n_components=64, covariance_type="diag")
    gmm.fit(xx)
    N = xx.shape[0]
    Q = gmm.predict_proba(xx)
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    d_pi = Q_sum.squeeze(), gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        -Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covariances_
        + 2 * Q_xx * gmm.means_
    )
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))


def build_spatial_pyramid(image, descriptor, level):
    """
    Rebuild the descriptors according to the level of pyramid
    """
    labels = kmeans.predict(descriptor)
    positions = locate_descriptor(image, descriptor)
    bh, bw = 2 ** (3 - level), 2 ** (3 - level)

    pyramid = []
    # for
    #     for i in range(bh):
    #         for j in range(bw):
    #             pyramid.append(np.bincount())
    return pyramid


def spatial_pyramid_matching(image, descriptor, codebook, level):
    pyramid = []
    pyramid += build_spatial_pyramid(image, descriptor, level=0)
    pyramid += build_spatial_pyramid(image, descriptor, level=1)
    pyramid += build_spatial_pyramid(image, descriptor, level=2)
    code = [generate_feats(crop, codebook) for crop in pyramid]
    code_level_0 = 0.25 * np.asarray(code[0]).flatten()
    code_level_1 = 0.25 * np.asarray(code[1:5]).flatten()
    code_level_2 = 0.5 * np.asarray(code[5:]).flatten()
    return np.concatenate((code_level_0, code_level_1, code_level_2))
