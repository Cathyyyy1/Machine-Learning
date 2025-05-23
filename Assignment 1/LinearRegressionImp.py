import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def fit_LinRegr(X_train, y_train):
    """
    This function computes the parameters w of a linear plane which best fits
    the training dataset.

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        the ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature. Each element in y_train takes a Real value.
    
    Returns
    -------
    w: numpy.ndarray with shape (d+1,)
        represents the coefficients of the line computed by the pocket
        algorithm that best separates the two classes of training data points.
        The dimensions of this vector is (d+1) as the offset term is accounted
        in the computation.
    """
    
    row, col = X_train.shape
    # augment X_train
    x_aug = np.hstack((np.ones(shape=(row, 1)), X_train))
    # Compute the pseudo-inverse of x_aug
    x_aug_trans = np.transpose(x_aug)
    x_aug_trans_dot_x_aug = np.dot(x_aug_trans, x_aug)
    x_aug_trans_dot_x_aug_inv = np.linalg.pinv(x_aug_trans_dot_x_aug)
    x_aug_trans_dot_x_aug_inv_dot_x_trans = np.dot(x_aug_trans_dot_x_aug_inv, x_aug_trans)
    # Compute weight least square
    w = np.dot(x_aug_trans_dot_x_aug_inv_dot_x_trans, y_train)
    
    return w


def mse(X_train, y_train, w):
    """
    This function finds the mean squared error introduced by the linear plane
    defined by w.

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        the ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature. Each element in y_train takes a Real value.
    w: numpy.ndarray with shape (d+1,)
        represents the coefficients of a linear plane.

    Returns
    -------
    avgError: float
        he mean squared error introduced by the linear plane defined by w.
    """
    
    # Accquire the length of data points and augment X_train
    N = len(X_train)
    row, col = X_train.shape
    x_aug = np.hstack((np.ones(shape=(row, 1)), X_train))
    
    # Loop through all the data point and compute squared error for each data point
    sum = 0
    for x in range(len(X_train)):
      prediction = pred(x_aug[x], w)
      sum += pow((y_train[x] - prediction), 2)
      
    # Compute averaged squared error
    avgError = sum / N
    
    return avgError


def pred(x_i, w):
    """
    This function finds finds the prediction by the classifier defined by w.

    Parameters
    ----------
    x_i: numpy.ndarray with shape (d+1,)
        Represents the feature vector of (d+1) dimensions of the ith test
        datapoint.
    w: numpy.ndarray with shape (d+1,)
        Represents the coefficients of a linear plane.
    
    Returns
    -------
    pred_i: int
        The predicted class.
    """

    pred_i = np.dot(x_i, w)
    return pred_i


def test_SciKit(X_train, X_test, y_train, y_test):
    """
    This function will output the mean squared error on the test set, which is
    obtained from the mean_squared_error function imported from sklearn.metrics
    library to report the performance of the model fitted using the 
    LinearRegression model available in the sklearn.linear_model library.
    
    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        Represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    X_test: numpy.ndarray with shape (M,d)
        Represents the matrix of input features where M is the total number of
        testing samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        The ith component represents output observed in the training set for the
        ith row in X_train matrix which corresponds to the ith input feature.
    y_test: numpy.ndarray with shape (M,)
        The ith component represents output observed in the test set for the
        ith row in X_test matrix which corresponds to the ith input feature.
    
    Returns
    -------
    error: float
        The mean squared error on the test set.
    """

    regressor = LinearRegression()
    
    # Run the fit function to train the model. 
    regressor.fit(X_train, y_train)
    
    # Use the predict function to perform predictions using the trained
    # model. 
    y_pred = regressor.predict(X_test)
    
    # Use the mean_squared_error function to find the mean squared error
    # on the test set. Don't forget to return the mean squared error.
    error = mean_squared_error(y_test, y_pred)
    return error


def subtestFn():
    """
    This function tests if your solution is robust against singular matrix.
    X_train has two perfectly correlated features.
    """

    X_train = np.asarray([[1, 2],
                          [2, 4],
                          [3, 6],
                          [4, 8]])
    y_train = np.asarray([1, 2, 3, 4])
    
    try:
      w = fit_LinRegr(X_train, y_train)
      print("weights: ", w)
      print("NO ERROR")
    except:
      print("ERROR")


def testFn_Part2():
    """
    This function loads diabetes dataset and splits it into train and test set.
    Then it finds and prints the mean squared error from your linear regression
    model and the one from the scikit library.
    """

    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
    
    w = fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e = mse(X_test, y_test, w)
    
    #Testing Part 2b
    scikit = test_SciKit(X_train, X_test, y_train, y_test)
    
    print(f"Mean squared error from Part 2a is {e}")
    print(f"Mean squared error from Part 2b is {scikit}")


if __name__ == "__main__":
    print (f"{12*'-'}subtestFn{12*'-'}")
    subtestFn()

    print (f"{12*'-'}testFn_Part2{12*'-'}")
    testFn_Part2()
