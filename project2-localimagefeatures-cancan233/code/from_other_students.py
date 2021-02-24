def get_features(image, x, y, feature_width):
    features = np.zeros((len(x), 128))

    grad_x = filters.sobel_h(image)
    grad_y = filters.sobel_v(image)
    assert grad_x.shape == grad_y.shape

    # compute gradient magnitudes and orientations
    grad_mags = np.sqrt(np.square(grad_x) + np.square(grad_y))
    grad_ors = np.arctan2(grad_y, grad_x)

    # bin pi and -pi together
    grad_ors[grad_ors == np.pi] = -np.pi
    assert grad_ors.shape == grad_mags.shape == grad_x.shape

    bins = [
        -np.pi,
        -3 / 4 * np.pi,
        -np.pi / 2,
        -np.pi / 4,
        0,
        np.pi / 4,
        np.pi / 2,
        3 / 4 * np.pi,
        np.pi,
    ]

    hw = feature_width // 2
    valid_points = list(range(len(x)))
    for i in range(len(x)):
        row = int(y[i])
        col = int(x[i])

        # check if feature is out of bounds
        if _is_grid_out(image.shape[0], image.shape[1], col, row, hw):
            valid_points.remove(i)
            continue

        # extract 4x4 cell patch
        ors = grad_ors[row - hw - 1 : row + hw - 1, col - hw - 1 : col + hw - 1]
        mags = grad_mags[row - hw - 1 : row + hw - 1, col - hw - 1 : col + hw - 1]
        assert ors.shape[0] == feature_width and ors.shape[1] == feature_width

        # reshape so that each square can be processed together
        ors = ors.reshape(feature_width // 4, 4, feature_width // 4, 4)
        mags = mags.reshape(feature_width // 4, 4, feature_width // 4, 4)

        # bin squares
        bin_indices = np.digitize(ors, bins) - 1

        # add orientation vectors to features
        l_mags = np.where(bin_indices == 0, mags, 0.0)
        features[i, 0::8] = l_mags.sum(3).sum(1).reshape(-1)

        l_mags = np.where(bin_indices == 1, mags, 0.0)
        features[i, 1::8] = l_mags.sum(3).sum(1).reshape(-1)

        l_mags = np.where(bin_indices == 2, mags, 0.0)
        features[i, 2::8] = l_mags.sum(3).sum(1).reshape(-1)

        l_mags = np.where(bin_indices == 3, mags, 0.0)
        features[i, 3::8] = l_mags.sum(3).sum(1).reshape(-1)

        l_mags = np.where(bin_indices == 4, mags, 0.0)
        features[i, 4::8] = l_mags.sum(3).sum(1).reshape(-1)

        l_mags = np.where(bin_indices == 5, mags, 0.0)
        features[i, 5::8] = l_mags.sum(3).sum(1).reshape(-1)

        l_mags = np.where(bin_indices == 6, mags, 0.0)
        features[i, 6::8] = l_mags.sum(3).sum(1).reshape(-1)

        l_mags = np.where(bin_indices == 7, mags, 0.0)
        features[i, 7::8] = l_mags.sum(3).sum(1).reshape(-1)

        # norm resultant vector
        norm = np.linalg.norm(features[i])
        features[i] *= 0.0 if norm == 0.0 else 1 / norm

    # take small power of features to improve performance
    return np.power(features[valid_points], 0.3)
