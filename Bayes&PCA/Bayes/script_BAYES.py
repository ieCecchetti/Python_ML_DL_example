from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_data_into_dataset(matrix, lable_array, id_class, url):
    # index to return
    offset = 0
    for filename in glob.glob(url):
        offset += 1
        img_data = np.asarray(Image.open(filename))
        # just store the whole matrix obtained by the 3-D to 1-D trasformation
        matrix.append(img_data.ravel())
        lable_array.append(id_class)
    return offset


def bayestest(X_matrix, Y_matrix, slice_size, seed):
    X_train, X_test, y_train, y_test = train_test_split(X_matrix,
                                                        Y_matrix,
                                                        test_size=slice_size,
                                                        random_state=seed)
    # print(X_train)
    # print(y_train)
    # print(X_test)
    # print(y_test)

    # Instantiate the classifier
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    # to_test = random.randint(0, len(X_test) - 1)
    # y_pred = gnb.predict([X_test[to_test]])
    y_pred = gnb.predict(X_test)
    # print("Classifier say is %s when it's %s" % (y_pred, y_test[to_test]))
    print("Number of mislabeled points out of a total %d is : %d"
          % (len(y_test), (y_test != y_pred).sum()))
    print("Naive Bayes right-labeling ratio is : %.2f %s"
          % ((len(y_test) - (y_test != y_pred).sum()) * 100 / (len(y_test)), "%"))
    print("-----------------------------------------------------")
    return (len(y_test) - (y_test != y_pred).sum()) * 100 / (len(y_test))


def main():
    right_labl_ratio = [0, 0, 0]
    lbl_vector = []
    X = []
    # image class offset in matrix
    di = gi = hi = pi = 0
    di = load_data_into_dataset(X, lbl_vector, 'dog', '../../res/PACS_homework/dog/*.jpg')
    gi = load_data_into_dataset(X, lbl_vector, 'guitar', '../../res/PACS_homework/guitar/*.jpg')
    hi = load_data_into_dataset(X, lbl_vector, 'house', '../../res/PACS_homework/house/*.jpg')
    pi = load_data_into_dataset(X, lbl_vector, 'person', '../../res/PACS_homework/person/*.jpg')
    # print(feature_matrix)
    print("CASE: Full-PC Matrix")
    right_labl_ratio[0] = bayestest(X, lbl_vector, 0.33, 10)

    # Repeat the splitting, training, and testing for the data projected onto
    # first two principal components,then third and fourth principal components.
    feature_matrix_t = PCA(2).fit_transform(X)
    print("CASE: 2-PC Matrix... (first 2)")
    right_labl_ratio[1] = bayestest(feature_matrix_t, lbl_vector, 0.33, 10)

    feature_matrix_t = PCA(4).fit_transform(X)
    print("CASE: 4-PC Matrix... (3rd and 4th)")
    right_labl_ratio[2] = bayestest(feature_matrix_t[:, 2:4], lbl_vector, 0.33, 10)

    n_groups = 3
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4

    rects1 = ax.bar(index, right_labl_ratio, bar_width,
                    alpha=opacity, color='b', label='Ratios')
    ax.set_xlabel('PCs')
    ax.set_ylabel('Ratios')
    ax.set_title('Scores per different PCs')
    ax.set_xticks(index)
    ax.set_xticklabels(('full', '1st&2nd', '3rd&4th'))
    ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()