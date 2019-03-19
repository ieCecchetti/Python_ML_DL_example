import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, KFold, cross_val_score

def main_function():
    # import some data to play with
    iris = datasets.load_iris()
    # we only take the first two features.
    X = iris.data[:, :2]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)
    # SVM regularization parameter
    C = [10**(-3), 10**(-2), 10**(-1), 1, 10**1, 10**2, 10**3]
    gamma = [10 ** (-9), 10 ** (-7), 10 ** (-5), 10 ** (-3)]

    plt.figure(figsize=(30, 20))

    min_mislabled = [10**(-3), 10**(-9), len(y_test)]
    for j in gamma:
        k = 1
        for i in C:
            k_fold = KFold(n_splits=5)
            scores = list()
            for train_indices, test_indices in k_fold.split(X_train):
                print("scale type C = %.3f and C = %f" % (i, j))
                clf = svm.SVC(kernel='rbf', C=i, gamma=j)
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)

                mislabled_num = (y_test != y_pred).sum()
                if mislabled_num < min_mislabled[2]:
                    min_mislabled[0] = i
                    min_mislabled[1] = j
                    min_mislabled[2] = mislabled_num

                print("Number of mislabeled points out of a total %d is : %d"
                      % (len(y_test), mislabled_num))
                print("SVM accuracy is : %.2f %s"
                      % ((len(y_test) - mislabled_num) * 100 / (len(y_test)), "%"))
                print("-----------------------------------------------------")
                k += 1
        plt.show()

    '''
    How do the boundaries change? Why?
    --
    The C parameter tells the SVM optimization how much you want to avoid misclassifying each training example. 
    For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a 
    better job of getting all the training points classified correctly. Conversely, a very small value of C will 
    cause the optimizer to look for a larger-margin separating hyperplane, even if that hyperplane misclassifies 
    more points. For very tiny values of C, you should get misclassified examples, often even if your training 
    data is linearly separable.
    
    The gamma parameter defines how far the influence of a single training example reaches, with low values meaning 
    ‘far’ and high values meaning ‘close’. Intuitively, a small gamma value define a Gaussian function with a large 
    variance. In this case, two points can be considered similar even if are far from each other. In the other hand, 
    a large gamma value means define a Gaussian function with a small variance and in this case, two points are 
    considered similar just if they are close to each other.
    '''

    print("scale type best C = %.3f gamma= %f" % (min_mislabled[0], min_mislabled[1]))
    clf = svm.SVC(kernel='rbf', C=min_mislabled[0], gamma=min_mislabled[1])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Number of mislabeled points out of a total %d is : %d"
          % (len(y_test), mislabled_num))
    print("SVM accuracy is : %.2f %s"
          % ((len(y_test) - mislabled_num) * 100 / (len(y_test)), "%"))
    print("-----------------------------------------------------")


if __name__ == '__main__':
    main_function()