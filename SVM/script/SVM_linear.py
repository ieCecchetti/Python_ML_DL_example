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
    gamma = [10 ** (-9), 10 ** (-2), 10 ** (-1), 1, 10 ** 1, 10 ** 2, 10 ** 3]
    mislabled = [0, 0, 0, 0, 0, 0, 0]
    mislabled_ratio = np.empty(shape=7, dtype=float)  # 3 element array

    plt.figure(figsize=(8, 8))
    k = 1
    min_mislabled = [10**(-3), len(y_validation)]
    for i in C:
        print("scale type C = %.3f" % (i,))
        clf = svm.SVC(kernel='linear', C=i)
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
        plt.title('SVC with linear kernel')

        y_pred = clf.predict(X_validation)
        # print("Classifier say is %s when it's %s" % (y_pred, y_test[to_test]))

        mislabled[k-1] = (y_validation != y_pred).sum()
        mislabled_ratio[k-1] = (len(y_validation)-mislabled[k-1])*100/(len(y_validation))

        # take count of the best storage for show at the end
        if mislabled[k-1] < min_mislabled[1]:
            min_mislabled[0] = i
            min_mislabled[1] = mislabled[k-1]

        print("Number of mislabeled points out of a total %d is : %d"
              % (len(y_validation), mislabled[k-1]))
        print("SVM accuracy is : %.2f %s"
              % ((len(y_validation) - mislabled[k-1]) * 100 / (len(y_validation)), "%"))
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

    print("scale type best C = %.3f" % (min_mislabled[0],))
    clf = svm.SVC(kernel='linear', C=min_mislabled[0])
    clf.fit(X_train, y_train)

    # show result through graph
    n_groups = 7

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, mislabled, bar_width,
                    alpha=opacity, color='b',
                    label='n_success')
    rects2 = ax.bar(index + bar_width, mislabled_ratio, bar_width,
                    alpha=opacity, color='r',
                    label='percentage')

    ax.set_xlabel('C')
    ax.set_ylabel('miss/ratio')
    ax.set_title('Scores by C value [linear kernel]')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('10^(-3)', '10^(-2)', '10^(-1)', '1', '10^1', '10^2', '10^3'))
    ax.legend()

    fig.tight_layout()
    plt.show()

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
    plt.title('SVC with linear kernel')

    y_pred = clf.predict(X_test)
    mislabled_num = (y_test != y_pred).sum()
    # print("Classifier say is %s when it's %s" % (y_pred, y_test[to_test]))

    print("Number of mislabeled points out of a total %d is : %d"
          % (len(y_test), mislabled_num))
    print("SVM accuracy is : %.2f %s"
          % ((len(y_test) - mislabled_num) * 100 / (len(y_test)), "%"))
    print("-----------------------------------------------------")


if __name__ == '__main__':
    main_function()