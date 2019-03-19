from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import glob
import random


def load_data_into_dataset(matrix, url):
    # index to return
    offset = 0
    for filename in glob.glob(url):
        offset += 1
        img_data = np.asarray(Image.open(filename))
        # just store the whole matrix obtained by the 3-D to 1-D trasformation
        matrix.append(img_data.ravel())
    return offset


'''
example of use:
plot_graph('First 6 PCs', '1st PC', '2nd PC', X_fsix, selected_img, 'b')
'''


def plot_graph(title, x_axis_lbl, y_axis_lbl, pca_matrix, image_id, color, **kwargs):
    frm = kwargs.get('frm', 0)
    to = kwargs.get('to', 0)
    # shape is a property of both numpy ndarray's and matrices
    # will return a tuple (m, n), where m is the number of rows, and n is the number of columns.
    if frm is 0 and to is 0:
        n_components = pca_matrix.shape[1]
        for i in range(0, n_components, +2):
            plt.scatter(pca_matrix[image_id, i],
                        pca_matrix[image_id, i + 1],
                        c=color)
    else:
        for i in range(int(frm)-1, int(to)-1, +2):
            plt.scatter(pca_matrix[image_id, i],
                        pca_matrix[image_id, i + 1],
                        c=color)

    plt.title(title)
    plt.xlabel(x_axis_lbl)
    plt.ylabel(y_axis_lbl)


def plot_multiclass_graph(title, x_axis_lbl, y_axis_lbl, pca_matrix, class_lable, class_dims):

    last_dim = 0
    for i in range(0, len(class_lable)):
        plt.scatter(pca_matrix[range(last_dim, last_dim+class_dims[i]), 0],
                    pca_matrix[range(last_dim, last_dim+class_dims[i]), 1],
                    c=class_lable[i])
        last_dim += class_dims[i]
    plt.title(title)
    plt.xlabel(x_axis_lbl)
    plt.ylabel(y_axis_lbl)


def inverse_image_plot(scatter_matrix, image_id, pca_matrix, std_matrix):
    # recompute inverse projection
    n_components = pca_matrix.shape[1]
    pca = PCA(n_components)
    X_t = pca.fit_transform(std_matrix)
    approx = pca.inverse_transform(X_t)
    approx = (approx * np.std(scatter_matrix)) + np.mean(scatter_matrix)
    return approx[image_id]


def pca_from_scratch(n_components, first_or_last, std_matrix):
    # PCA(last 6)
    # Obtain the Eigenvectors and Eigenvalues from the covariance matrix or
    # correlation matrix, or perform Singular Vector Decomposition.
    cov_X = np.cov(std_matrix)
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_X)

    # Raises an AssertionError if two objects are not equal up to desired precision.
    # The test verifies identical shapes and that the elements of actual and desired satisfy.
    for ev in eig_vec_cov:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    # reverse=False == Ascending, so from the one with highest variance to minimum
    if first_or_last is True:
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
    else:
        eig_pairs.sort(key=lambda x: x[0], reverse=False)

    matrix_w = np.hstack([eig_pairs[i][1].reshape(1087, 1) for i in range(n_components)])
    return matrix_w


def pca_scratch(scatter_matrix, id_image, pca_matrix, std_matrix):
    # recompute inverse projection
    n_component = pca_matrix.shape[1]
    pca = PCA(6)
    pca.fit_transform(std_matrix)
    approx = pca.inverse_transform(pca_matrix)
    approx = (approx * np.std(scatter_matrix)) + np.mean(scatter_matrix)
    return approx[id_image]


def main():
    X = []
    # image class offset in matrix
    di = gi = hi = pi = 0
    di = load_data_into_dataset(X, '../../res/PACS_homework/dog/*.jpg')
    gi = load_data_into_dataset(X, '../../res/PACS_homework/guitar/*.jpg')
    hi = load_data_into_dataset(X, '../../res/PACS_homework/house/*.jpg')
    pi = load_data_into_dataset(X, '../../res/PACS_homework/person/*.jpg')

    lbl_vector = ['b', 'r', 'g', 'c']

    # standardization of feature matrix
    # makes feature_matrix with mean 0 & variance 1 -> as definition od standardization
    std_X = (X - np.mean(X)) / np.std(X)

    selected_img = random.randint(0, di+gi+hi+pi-1)
    # selected_img = 0

    plt.figure(figsize=(8, 8))

    # PCA(2)
    X_two = PCA(2).fit_transform(std_X)
    print("[!] PCA(2) Successfully computed.")

    # PCA(6)
    X_fsix = PCA(6).fit_transform(std_X)
    print("[!] PCA(6) Successfully computed.")

    # PCA(60)
    X_sixty = PCA(60).fit_transform(std_X)
    print("[!] PCA(60) Successfully computed.")

    # Using scatter-plot, visualize the dataset projected on first 2 PC.
    # Repeat the exercise with only 3&4 PC, and with 10&11.

    # Plot by using first 2 Principle Components
    plt.subplot(221)
    plt.scatter(X_sixty[:189, 0], X_sixty[:189, 1], c='b', alpha=0.7, edgecolors='none', label='dog')
    plt.scatter(X_sixty[189:620, 0], X_sixty[189:620, 1], c='r', alpha=0.7, edgecolors='none', label='person')
    plt.scatter(X_sixty[620:807, 0], X_sixty[620:807, 1], c='g', alpha=0.7, edgecolors='none', label='guitar')
    plt.scatter(X_sixty[807:, 0], X_sixty[807:, 1], c='c', alpha=0.7, edgecolors='none', label='house')
    plt.title("1stPC & 2ndPC Comparison")
    plt.xlabel("1st PC")
    plt.ylabel("2nd PC")
    plt.legend()
    # Plot by using 3th and 4th Principle Components
    plt.subplot(222)
    plt.scatter(X_sixty[:189, 2], X_sixty[:189, 3], c='b', alpha=0.7, edgecolors='none', label='dog')
    plt.scatter(X_sixty[189:620, 2], X_sixty[189:620, 3], c='r', alpha=0.7, edgecolors='none', label='person')
    plt.scatter(X_sixty[620:807, 2], X_sixty[620:807, 3], c='g', alpha=0.7, edgecolors='none', label='guitar')
    plt.scatter(X_sixty[807:, 2], X_sixty[807:, 3], c='c', alpha=0.7, edgecolors='none', label='house')
    plt.title("3rdPC & 4thPC Comparison")
    plt.xlabel("3rd PC")
    plt.ylabel("4th PC")
    plt.legend()
    # Plot by using 3th and 4th Principle Components
    plt.subplot(223)
    plt.scatter(X_sixty[:189, 9], X_sixty[:189, 10], c='b', alpha=0.7, edgecolors='none', label='dog')
    plt.scatter(X_sixty[189:620, 9], X_sixty[189:620, 10], c='r', alpha=0.7, edgecolors='none', label='person')
    plt.scatter(X_sixty[620:807, 9], X_sixty[620:807, 10], c='g', alpha=0.7, edgecolors='none', label='guitar')
    plt.scatter(X_sixty[807:, 9], X_sixty[807:, 10], c='c', alpha=0.7, edgecolors='none', label='house')
    plt.title("10thPC & 11thPC Comparison")
    plt.xlabel("10th PC")
    plt.ylabel("11th PC")
    plt.legend()

    # PCA(last 6)
    X_lsix = pca_from_scratch(6, False, std_X)
    print("[!] PCA(last6) Successfully computed.")

    # Visualize X t using scatter-plot with different colors standing for different classes
    plt.subplot(224)
    plot_multiclass_graph('Multiclass Graph 2PCs', '1st PC', '2nd PC',
                          X_two, lbl_vector, [di, gi, hi, pi])
    plt.subplots_adjust(bottom=0.5, right=0.4, top=0.9)
    plt.show()

    # comulative variance of PCs to determs how much Pcs i should use to maximize
    # the recognition
    pca1 = PCA().fit(std_X)
    plt.plot(np.cumsum(pca1.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    # Choose one image and shows what happens to the image when you re-project it with only
    # first 60 PC, first 6 PC, first 2 PC, last 6 PC. [not in the exact order]
    original = np.reshape(X[selected_img], (227, 227, 3)).astype(int)
    plt.imshow(original, interpolation='nearest')
    plt.title('Full 1087 PCs')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(8, 8))

    to_print = inverse_image_plot(X, selected_img, X_two, std_X)
    plt.subplot(221)
    image = np.reshape(to_print, (227, 227, 3)).astype(int)
    plt.imshow(image, interpolation='nearest')
    plt.title('First 2PCs')
    plt.axis('off')

    to_print = inverse_image_plot(X, selected_img, X_fsix, std_X)
    plt.subplot(222)
    image = np.reshape(to_print, (227, 227, 3)).astype(int)
    plt.imshow(image, interpolation='nearest')
    plt.title('First 6 PCs')
    plt.axis('off')

    to_print = inverse_image_plot(X, selected_img, X_sixty, std_X)
    plt.subplot(223)
    image = np.reshape(to_print, (227, 227, 3)).astype(int)
    plt.imshow(image, interpolation='nearest')
    plt.title('First 60 PCs')
    plt.axis('off')

    to_print = pca_scratch(X, selected_img, X_lsix, std_X)
    plt.subplot(224)
    image = np.reshape(to_print, (227, 227, 3)).astype(int)
    plt.imshow(image, interpolation='nearest')
    plt.title('Last 6 PCs')
    plt.axis('off')

    plt.subplots_adjust(bottom=0.5, right=0.4, top=0.9)

    # print all subplots created
    plt.show()


if __name__ == '__main__':
    main()



