# A set of helpers which are useful for debugging SIFT!
# Feel free to take a look around in case you are curious,
# but you shouldn't need to know exactly what goes on,
# and you certainly don't need to change anything

import scipy.io as scio
import skimage
import numpy as np
import visualize
import matplotlib.pyplot as plt
import math

# Gives you the TA solution for the interest points you
# should find
def cheat_interest_points(eval_file, scale_factor):

	file_contents = scio.loadmat(eval_file)

	x1 = file_contents['x1']
	y1 = file_contents['y1']
	x2 = file_contents['x2']
	y2 = file_contents['y2']

	x1 = x1 * scale_factor
	y1 = y1 * scale_factor
	x2 = x2 * scale_factor
	y2 = y2 * scale_factor

	x1 = x1.reshape(-1)
	y1 = y1.reshape(-1)
	x2 = x2.reshape(-1)
	y2 = y2.reshape(-1)

	return x1, y1, x2, y2

def estimate_fundamental_matrix(ground_truth_correspondence_file):
    F_path = ground_truth_correspondence_file[:-3] + 'npy'
    return np.load(F_path)

def evaluate_correspondence(img_A, img_B, ground_truth_correspondence_file,
	scale_factor, x1_est, y1_est, x2_est, y2_est, matches, confidences, vis, filename="notre_dame_matches.jpg"):

	# 'unscale' interest points to compare with ground truth points
	x1_est_scaled = x1_est / scale_factor
	y1_est_scaled = y1_est / scale_factor
	x2_est_scaled = x2_est / scale_factor
	y2_est_scaled = y2_est / scale_factor

	conf_indices = np.argsort(-confidences, kind='mergesort')
	matches = matches[conf_indices,:]
	confidences = confidences[conf_indices]

	# we want to see how good our matches are, extract the coordinates of each matched
	# point

	x1_matches = np.zeros(matches.shape[0])
	y1_matches = np.zeros(matches.shape[0])
	x2_matches = np.zeros(matches.shape[0])
	y2_matches = np.zeros(matches.shape[0])

	for i in range(matches.shape[0]):

		x1_matches[i] = x1_est_scaled[int(matches[i, 0])]
		y1_matches[i] = y1_est_scaled[int(matches[i, 0])]
		x2_matches[i] = x2_est_scaled[int(matches[i, 1])]
		y2_matches[i] = y2_est_scaled[int(matches[i, 1])]

	good_matches = np.zeros((matches.shape[0]), dtype=np.bool)

	# Loads `ground truth' positions x1, y1, x2, y2
	file_contents = scio.loadmat(ground_truth_correspondence_file)

	# x1, y1, x2, y2 = scio.loadmat(eval_file)
	x1 = file_contents['x1']
	y1 = file_contents['y1']
	x2 = file_contents['x2']
	y2 = file_contents['y2']

	pointsA = np.zeros((len(x1), 2))
	pointsB = np.zeros((len(x2), 2))

	for i in range(len(x1)):
		pointsA[i, 0] = x1[i]
		pointsA[i, 1] = y1[i]
		pointsB[i, 0] = x2[i]
		pointsB[i, 1] = y2[i]

	correct_matches = 0

	F = estimate_fundamental_matrix(ground_truth_correspondence_file)
	top50 = 0
	top100 = 0

	for i in range(x1_matches.shape[0]):
		pointA = np.ones((1, 3))
		pointB = np.ones((1, 3))
		pointA[0,0] = x1_matches[i]
		pointA[0,1] = y1_matches[i]
		pointB[0,0] = x2_matches[i]
		pointB[0,1] = y2_matches[i]


		if abs(pointB @ F @ np.transpose(pointA)) < .1:
			x_dists = x1 - x1_matches[i]
			y_dists = y1 - y1_matches[i]

			# computes distances of each interest point to the ground truth point
			dists = np.sqrt(np.power(x_dists, 2.0) + np.power(y_dists, 2.0))
			closest_ground_truth = np.argmin(dists, axis=0)
			offset_x1 = x1_matches[i] - x1[closest_ground_truth]
			offset_y1 = y1_matches[i] - y1[closest_ground_truth]
			offset_x1 *= img_B.shape[0] / img_A.shape[0]
			offset_y1 *= img_B.shape[0] / img_A.shape[0]
			offset_x2 = x2_matches[i] - x2[closest_ground_truth]
			offset_y2 = y2_matches[i] - y2[closest_ground_truth]
			offset_dist = np.sqrt(np.power(offset_x1 - offset_x2, 2) + np.power(offset_y1 - offset_y2, 2))
			if offset_dist < 70:
				correct_matches += 1
				good_matches[i] = True
		if i == 49:
			print(f'Accuracy on 50 most confident: {int(100 * correct_matches / 50)}%')
			top50 = correct_matches
		if i == 99:
			print(f'Accuracy on 100 most confident: {int(100 * correct_matches / 100)}%')
			top100 = correct_matches

	print(f'Accuracy on all matches: {int(100 * correct_matches / len(matches))}%')

	if vis > 0:
		print("Vizualizing...")
		visualize.show_correspondences(img_A, img_B, x1_est / scale_factor, y1_est / scale_factor, x2_est / scale_factor, y2_est / scale_factor, matches, good_matches, vis, filename)

	return top50, top100, correct_matches
