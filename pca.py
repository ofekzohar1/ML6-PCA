import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people


def plot_vector_as_image(image, h, w, title):
	"""
	utility function to plot a vector as image.
	Args:
	image - vector of pixels
	h, w - dimensions of original pi
	"""
	plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
	plt.title(title, size=12)  # Give meaningful title
	plt.show()


def get_pictures_by_name(name='Ariel Sharon'):
	"""
	Given a name returns all the pictures of the person with this specific name.
	YOU CAN CHANGE THIS FUNCTION!
	THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
	"""
	lfw_people = load_data()
	selected_images = []
	n_samples, h, w = lfw_people.images.shape
	target_label = list(lfw_people.target_names).index(name)
	for image, target in zip(lfw_people.images, lfw_people.target):
		if (target == target_label):
			image_vector = image.reshape((h*w, 1))
			selected_images.append(image_vector)
	return selected_images, h, w


def load_data():
	# Don't change the resize factor!!!
	lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
	return lfw_people

######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""


def PCA(X, k):
	"""
	Compute PCA on the given matrix.

	Args:
		X - Matrix of dimensions (n,d). Where n is the number of sample points and d is the dimension of each sample.
			For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimension of the
			matrix would be (10,100).
		k - number of eigenvectors to return

	Returns:
		U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the
			largest k eigenvectors of the covariance matrix.
		S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
	"""
	n = len(X)

	# Shift the observations so the mean would be 0 by subtract the original mean
	avg = np.mean(X, axis=0)
	X -= avg

	sigma = (np.transpose(X) @ X) / n  # Empirical COV matrix
	eigvals, eigvecs = np.linalg.eig(sigma)

	idx = eigvals.argsort()[-k:][::-1]  # Get the index of the k largest eigenvalues in DES order
	S = eigvals[idx]  # k largest eigvals
	U = np.transpose(eigvecs)[idx]  # k largest eigvecs (correspond to eigvals)

	return U, S


def part_b(k, name=''):
	selected_images, h, w = get_pictures_by_name(name) if name != '' else get_pictures_by_name()
	X = np.array(selected_images)[:, :, 0]  # Picture as rows
	U, S = PCA(X, k)  # Perform PCA
	for i, u in enumerate(U):  # Plot eigenvector images
		plot_vector_as_image(u, h, w, f"Eigenvector {i+1}")


def part_c(K, name=''):
	selected_images, h, w = get_pictures_by_name(name) if name != '' else get_pictures_by_name()
	X = np.array(selected_images)[:, :, 0]  # Picture as rows
	n = len(X)
	rand = np.random.choice(n, 5, replace=False)  # Choose 5 random obs.
	for i in rand:  # plot the original images
		plot_vector_as_image(X[i], h, w, f"Original - image {i + 1}")

	sum_l2s = []
	for k in K:
		U, S = PCA(X, k)  # Perform PCA to k-dim
		X_hat = np.transpose(np.transpose(U) @ U @ np.transpose(X))  # perform encode-decode procedure. image as rows
		norms = np.linalg.norm(X - X_hat, axis=0) ** 2  # calculate ||xi-xi_hat||^2 for every i
		sum_l2s.append(np.sum(norms))  # sum SQ all the l2 norms
		for i in rand:  # plot the encode-decode according to k-dim images
			plot_vector_as_image(X_hat[i], h, w, f"Projection to {k} dim Encode-Decode image {i+1}")

	# Plot the l2 sums as a function of the dim k
	plt.clf()
	plt.plot(K, sum_l2s)
	plt.xlabel("k")
	plt.ylabel("Sum squared of l2 norms")
	plt.show()


def main():  # Uncomment to perform the desired operation
	#part_b(10)
	K = [1, 5, 10, 30, 50, 100]
	#part_c(K)


if __name__ == "__main__":
	main()
