import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

def main_function():
    # import some data to play with
    iris = datasets.load_iris()
    # we only take the first two features.
    X = iris.data[:, :2]
    y = iris.target
    X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                        y,
                                                        test_size=0.5,
                                                        random_state=42)
    X_validation, X_test, y_validation, y_test = train_test_split(X_temp,
                                                                  y_temp,
                                                                  test_size=0.4,
                                                                  random_state=42)
    # SVM regularization parameter
    C = [10**(-3), 10**(-2), 10**(-1), 1, 10**1, 10**2, 10**3]
    gamma = [10 ** (-9), 10 ** (-7), 10 ** (-5), 10 ** (-3)]

    plt.figure(figsize=(8, 8))

    min_mislabled = [10**(-3), 10**(-9), len(y_validation)]
    for j in gamma:
        k = 1
        for i in C:
            print("scale type C = %.3f and C = %f" % (i, j))
            clf = svm.SVC(kernel='rbf', C=i, gamma=j)
            clf.fit(X_train, y_train)

            # create a mesh to plot in
            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
            h = (x_max / x_min) / 100
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))

            plt.subplot(3, 3, k)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

            plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)
            plt.xlabel('Sepal length')
            plt.ylabel('Sepal width')
            plt.xlim(xx.min(), xx.max())
            plt.title('gamma= '+str(j)+' C= '+str(i))

            y_pred = clf.predict(X_validation)
            # print("Classifier say is %s when it's %s" % (y_pred, y_test[to_test]))

            mislabled_num = (y_validation != y_pred).sum()
            if mislabled_num < min_mislabled[2]:
                min_mislabled[0] = i
                min_mislabled[1] = j
                min_mislabled[2] = mislabled_num

            print("Number of mislabeled points out of a total %d is : %d"
                  % (len(y_validation), mislabled_num))
            print("SVM accuracy is : %.2f %s"
                  % ((len(y_validation) - mislabled_num) * 100 / (len(y_validation)), "%"))
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
    '''

    print("scale type best C = %.3f gamma= %f" % (min_mislabled[0], min_mislabled[1]))
    clf = svm.SVC(kernel='rbf', C=min_mislabled[0], gamma=min_mislabled[1])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    mislabled_num = (y_test != y_pred).sum()
    print("Number of mislabeled points out of a total %d is : %d"
          % (len(y_test), mislabled_num))
    print("SVM accuracy is : %.2f %s"
          % ((len(y_test) - mislabled_num) * 100 / (len(y_test)), "%"))
    print("-----------------------------------------------------")


if __name__ == '__main__':
    main_function()