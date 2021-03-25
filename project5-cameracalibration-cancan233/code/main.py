import argparse
import os
from skimage import io
import numpy as np

from helpers import show_reprojections, get_matches, show_point_cloud, \
    get_markers, show_matches
import student


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Project 5 camera calibration!")
    parser.add_argument(
        '--sequence',
        required=True,
        choices=['mikeandikes', 'cards', 'dollar', 'extracredit'],
        help='Which image sequence to use')
    parser.add_argument(
        '--data',
        default=os.getcwd() + '/../data/',
        help='Location where your data is stored')
    parser.add_argument(
        '--ransac-iters',
        type=int,
        default=100,
        help='Number of samples to try in RANSAC')
    parser.add_argument(
        '--num-keypoints',
        type=int,
        default=5000,
        help='Number of keypoints to detect with ORB')
    parser.add_argument(
        '--no-intermediate-vis',
        action='store_true',
        help='Disables intermediate visualizations'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = os.path.join(args.data, args.sequence)
    image_files = os.listdir(data_dir)

    print(f'Loading {len(image_files)} images for {args.sequence} sequence...')
    images = []
    for image_file in image_files:
        images.append(io.imread(os.path.join(data_dir, image_file)))

    markers = get_markers(os.path.join(args.data, "markers.txt"))

    print('Calculating projection matrices...')
    Ms = [student.calculate_projection_matrix(image, markers) for image in images]

    if not args.no_intermediate_vis:
        show_reprojections(images, Ms, markers)

    points3d = []
    points3d_color = []

    for i in range(len(images) - 1):
        image1 = images[i]
        M1 = Ms[i]
        image2 = images[i + 1]
        M2 = Ms[i + 1]

        print(f'Getting matches for images {i + 1} and {i + 2} of {len(images)}...')
        points1, points2 = get_matches(image1, image2, args.num_keypoints)
        if not args.no_intermediate_vis:
            show_matches(image1, image2, points1, points2)

        print(f'Filtering with RANSAC...')
        F, inliers1, inliers2 = student.ransac_fundamental_matrix(
            points1, points2, args.ransac_iters)
        if not args.no_intermediate_vis:
            show_matches(image1, image2, inliers1, inliers2)

        print('Calculating 3D points for accepted matches...')
        points3d += student.matches_to_3d(inliers1, inliers2, M1, M2)
        points3d_color += [tuple(image1[int(point[1]), int(point[0]), :] / 255.0) for point in inliers1]

    for key in markers:
        points3d += markers[key]
        points3d_color += [(0, 0, 0)] * 4

    i = 0
    while i < len(points3d):
        x, y, z = points3d[i]
        if x < 0 or x > 7 or y < 0 or y > 7 or z < 0 or z > 7:
            points3d.pop(i)
            points3d_color.pop(i)
            i -= 1
        i += 1

    points3d = np.array(points3d)

    show_point_cloud(points3d, points3d_color)


if __name__ == '__main__':
    main()
