import os
import numpy as np
import matplotlib.pyplot as plt

def split_at_char(s, first, last ):
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""

def get_average_dist_to_origin(points):
    dist = (points - [0,0])**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)

    return np.mean(dist)

def read_data(folder, file):
    """
    Returns a list of lists containing the points of the
    reference and warped images.
    """

    print("Reading original and warped image datapoints...")

    with open(os.path.join("boat", "homography.txt"), "r") as file:
        image_data = [line for line in file]

    image1 = image_data[0]
    image1 = split_at_char(image1, "[", "]").split(";")
    image1 = [elem.split(",") for elem in image1]
    image1 = [list(map(int,i)) for i in image1]

    image2 = image_data[1]
    image2 = split_at_char(image2, "[", "]").split(";")
    image2 = [elem.split(",") for elem in image2]
    image2 = [list(map(int,i)) for i in image2]

    return image1, image2

def normalize_image_points(image):
    """
    Input: 2D list with x,y image points
    Output:
    """

    print()
    print("Normalizing data using similarity matrix...")

    image = np.array(image)
    mean, std = np.mean(image, 0), np.std(image)

    # define similarity transformation
    # no rotation, scaling using sdv and setting centroid as origin
    Transformation = np.array([[std/np.sqrt(2), 0, mean[0]],
                               [0, std/np.sqrt(2), mean[1]],
                               [0,   0, 1]])

    # apply transformation on data points
    Transformation = np.linalg.inv(Transformation)
    image = np.dot(Transformation, np.concatenate((image.T, np.ones((1, image.shape[0])))))

    # retrieve normalized image in the original input shape (25, 2)
    image = image[0:2].T

    print("translated origin:", np.mean(image, axis=0))
    print("average distance to origin:", get_average_dist_to_origin(image))

    return image, Transformation

def compute_matrix_A(points1, points2, no_points):
    """
    Input: Normalized correspondences for image1 and image2
    Output: Matrix A as defined in Zisserman p. 91
    """

    A = []

    for i in range(0, no_points):
        x, y = points1[i, 0], points1[i, 1]
        x_prime, y_prime = points2[i, 0], points2[i, 1]

        # create A_i according to the eq. in the book
        # here we are assuming w_i is one
        A.append([0, 0, 0, -x, -y, -1, y_prime*x, y_prime*y, y_prime])
        A.append([x, y, 1, 0, 0, 0, -x_prime*x, -x_prime*y, -x_prime])

    print()
    print("Stacked matrix A shape:", np.shape(A))

    return np.asarray(A)

def compute_SVD(matrix_A):
    print()
    print("Computing SVD...")

    return np.linalg.svd(matrix_A)

def get_vector_h(matrix_V):
    """
    Input: Matrix V from SVD of A
    Output: Unitary vector h (last column of V matrix of SVD)
    """
    print()
    print("Obtaining vector h...")

    h = matrix_V[-1,:] / matrix_V[-1,-1]

    return h

def main():
    # read image data points
    image1, image2 = read_data("boat", "homography.txt")

    # set data points to numpy arrays
    image1 = np.array(image1)
    image2 = np.array(image2)
    no_points = image1.shape[0]

    # normalize data
    image1_normalized, T = normalize_image_points(image1)
    image2_normalized, T_prime = normalize_image_points(image2)

    # get matrix A for each normalized correspondence (dims 2*n x 9)
    A = compute_matrix_A(image1_normalized, image2_normalized, no_points)

    # compute SVD of A
    U, S, V = compute_SVD(A)

    # get last column of V and normalize it (this is the vector h)
    h = get_vector_h(V)

    # obtain homography (H tilde)
    print()
    print("Reshaping to get homography H_tilde...")
    H_tilde = h.reshape(3,3)

    # denormalize to obtain homography (H) using the transformations and generalized pseudo-inverse
    H = np.dot(np.dot(np.linalg.pinv(T_prime), H_tilde), T)

    print()
    print("Denormalized to obtain homography H for 2D data points...")
    print("Matrix H:")
    print(H)

if __name__ == "__main__":
    main()
