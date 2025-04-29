import numpy as np
import random
import collections
import util

# YOU ARE NOT ALLOWED TO USE sklearn or Pytorch in this assignment


class Optimizer:

    def __init__(
        self, name, lr=0.001, gama=0.9, beta_m=0.9, beta_v=0.999, epsilon=1e-8
    ):
        # self.lr will be set as the learning rate that we use upon creating the object, i.e., lr
        # e.g., by creating an object with Optimizer("sgd", lr=0.0001), the self.lr will be set as 0.0001
        self.lr = lr

        # Based on the name used for creating an Optimizer object,
        # we set the self.optimize to be the desiarable method.
        if name == "sgd":
            self.optimize = self.sgd
        elif name == "heavyball_momentum":
            # setting the gamma parameter and initializing the momentum
            self.gama = gama
            self.v = 0
            self.optimize = self.heavyball_momentum
        elif name == "nestrov_momentum":
            # setting the gamma parameter and initializing the momentum
            self.gama = gama
            self.v = 0
            self.optimize = self.nestrov_momentum
        elif name == "adam":
            # setting beta_m, beta_v, and epsilon
            # (read the handout to see what these parametrs are)
            self.beta_m = beta_m
            self.beta_v = beta_v
            self.epsilon = epsilon

            # setting the initial first momentum of the gradient
            # (read the handout for more info)
            self.v = 0

            # setting the initial second momentum of the gradient
            # (read the handout for more info)
            self.m = 0

            # initializing the iteration number
            self.t = 1

            self.optimize = self.adam

    def sgd(self, gradient):
        # update the vector that will be used by gradient descent, which is learning rate * (- gradient)
        return self.lr * (-1) * gradient

    def heavyball_momentum(self, gradient):
        # Based on the euqation that momentum = - learning rate * gradient + momentum parameter * last momentum
        self.v = (-1) * self.lr * gradient + self.gama * self.v
        
        return self.v
        
    def nestrov_momentum(self, gradient):
        return self.heavyball_momentum(gradient)

    def adam(self, gradient):
        # first step, update m
        self.m = (1 - self.beta_m) * gradient + self.beta_m * self.m
        # update v
        self.v = (1 - self.beta_v) * np.square(gradient) + self.beta_v * self.v
        # compute m hat and v hat
        mHat = self.m / (1 - self.beta_m ** self.t)
        vHat = self.v / (1 - self.beta_v ** self.t)
        # Increament t by 1
        self.t += 1
        update = (-1) * (self.lr * mHat) / (np.sqrt(vHat) + self.epsilon)
        
        return update

class MultiClassLogisticRegression:
    def __init__(self, n_iter=10000, thres=1e-3):
        self.n_iter = n_iter
        self.thres = thres

    def fit(
        self,
        X,
        y,
        batch_size=64,
        lr=0.001,
        gama=0.9,
        beta_m=0.9,
        beta_v=0.999,
        epsilon=1e-8,
        rand_seed=4,
        verbose=False,
        optimizer="sgd",
    ):
        # setting the random state for consistency.
        np.random.seed(rand_seed)

        # find all classes in the train dataset.
        self.classes = self.unique_classes_(y)

        # assigning an integer value to each class, from 0 to (len(self.classes)-1)
        self.class_labels = self.class_labels_(self.classes)

        # one-hot-encode the labels.
        self.y_one_hot_encoded = self.one_hot(y)

        # add a column of 1 to the leftmost column.
        X = self.add_bias(X)

        # initialize the E_in list to keep track of E_in after each iteration.
        self.loss = []

        # initialize the weight parameters with a matrix of all zeros.
        # each row of self.weights contains the weights for one of the classes.
        self.weights = np.zeros(shape=(len(self.classes), X.shape[1]))

        # create an instance of optimizer
        opt = Optimizer(
            optimizer, lr=lr, gama=gama, beta_m=beta_m, beta_v=beta_v, epsilon=epsilon
        )

        i, update = 0, 0
        while i < self.n_iter:
            self.loss.append(
                self.cross_entropy(self.y_one_hot_encoded, self.predict_with_X_aug_(X))
            )

            # sample a batch of data, X_batch and y_batch, with batch_size number of datapoint uniformly at random
            chosen_Index = np.random.choice(X.shape[0], batch_size, replace=False)
            X_batch = X[chosen_Index]
            y_batch = self.y_one_hot_encoded[chosen_Index]
            
            # NOTE: for nestrov_momentum, the gradient is derived at a point different from self.weights
            # See the assignments handout or the lecture note for more information.

            # If the optimizer is nesterov_momentum
            # The weight we are deriving is (current weights + momentum_parameter * momentum)
            if optimizer == "nesterov_momentum":
                # Calculate the weight by adding the current momentum
                look_ahead_weights = self.weights + gama * opt.v
                gradient = self.compute_grad(X_batch, y_batch, look_ahead_weights)
            else:
                # Standard gradient calculation for other optimizers (SGD, heavy-ball momentum, adma)
                gradient = self.compute_grad(X_batch, y_batch, self.weights)
            
            # find the update vector by using the optimization method and update self.weights
            update = opt.optimize(gradient)
            self.weights += update
            
            # Stop criteria
            # Stop if the largest update is smaller than the threshold
            if np.max(np.abs(update)) < self.thres:
                break

            if i % 1000 == 0 and verbose:
                print(
                    " Training Accuray at {} iterations is {}".format(
                        i, self.evaluate_(X, self.y_one_hot_encoded)
                    )
                )
            i += 1
        return self

    def add_bias(self, X):
        # at index 0, insert 1, along axis=1 (column)
        X_biased = np.insert(X, 0, 1, axis=1)
        
        return X_biased
        
    def unique_classes_(self, y):
        return np.unique(y)

    def class_labels_(self, classes):
        # Initialize the dictionary
        label_dict = dict()
        for x in range(len(classes)):
            # add pair of label in the class and its index
            label_dict[classes[x]] = x
        
        return label_dict
        
    def one_hot(self, y):
        # Create a one-hot matrix of zeros with size of (number of output, number of classes)
        one_hot_matrix = np.zeros((len(y), len(self.class_labels)))
        
        # Map each label in y to its index and create one-hot encoding
        for index, label in enumerate(y):
            # index of each label from class_lables dictionary
            class_index = self.class_labels[label]
            # map the label to its index [index, class_index] to 1
            one_hot_matrix[index, class_index] = 1
        
        return one_hot_matrix

    def softmax(self, z):
        # softmax is calculated by each exp(z) / sum of total exp of elements in z
        z_exp = np.exp(z)
        # sum over the row and keep the dimension of matrix for further matrix division calculations
        sum_exp_z = np.sum(z_exp, axis=1,  keepdims=True)
        # Apply the softmax formula
        softmax = z_exp / sum_exp_z
        
        return softmax

    def predict_with_X_aug_(self, X_aug):
        # first compute the weight tranpose * X_aug
        wT_X_aug = np.dot(X_aug, np.transpose(self.weights))
        # apply softmax to wT_X_aug get the probability distribution matrix
        prediction = self.softmax(wT_X_aug)
        
        return prediction
        
    def predict(self, X):
        # augment X to X_aug
        X_aug = self.add_bias(X)
        # apply predict_with_X_aug_ to get probability distribution matrix
        prediction = self.predict_with_X_aug_(X_aug)
        
        return prediction
        
    def predict_classes(self, X):
        # apply predict to input matrix X, it augments X in predict and returns a probability matrix
        probability_matrix = self.predict(X)
        # apply argmax to each row to find the index of most possible class
        predicted_class_index = np.argmax(probability_matrix, axis=1)
        # convert index to the actual unique class label and return as numpy array
        predicted_class = np.array([self.classes[index] for index in predicted_class_index])
            
        return predicted_class

    def score(self, X, y):
        # plug in input X to get the predicted class array
        predicted_class = self.predict_classes(X)
        # compute the number of correctly labeled class
        correctly_labeled = 0
        for index, label in enumerate(y):
            if predicted_class[index] == label:
                correctly_labeled += 1
                
        # return the ratio of correct label with respect to total labels
        return correctly_labeled / len(y)
        
    def evaluate_(self, X_aug, y_one_hot_encoded):
        # Apply predict_with_X_aug_ to input datapoints to get the probability matrix
        prediction = self.predict_with_X_aug_(X_aug)
        # Get the index with max probability for each datapoint, which is the predicted class
        predicted_classes = np.argmax(prediction, axis=1)
        # Get the index of true labels for each datapoint
        true_labels = np.argmax(y_one_hot_encoded, axis=1)
        
        # Compute the total number of labels that are correctly labeled
        correctly_labeled = 0
        for index, predicted_label in enumerate(predicted_classes):
            if true_labels[index] == predicted_label:
                correctly_labeled += 1
        
        return correctly_labeled / len(true_labels)
        
    def cross_entropy(self, y_one_hot_encoded, probs):
        # Compute the log of probability and times it with y_one_hot_encoded
        cross_entropy_matrix = y_one_hot_encoded * np.log(probs)
        # Sum over the matrix to sum up the total cross-entropy loss
        total_loss = 0
        for i in range(y_one_hot_encoded.shape[0]):  # Loop over each data point
            for j in range(y_one_hot_encoded.shape[1]):  # Loop over each class for the current data point
                total_loss += cross_entropy_matrix[i, j]
        
        # Divide the total entropy by number of datapoints
        cross_entropy_loss = (-1) * total_loss / len(y_one_hot_encoded)
        
        return cross_entropy_loss
        
    def compute_grad(self, X_aug, y_one_hot_encoded, w):
        # Compute wT time X_aug and plug in to softmax to get the probability matrix
        z = np.dot(X_aug, np.transpose(w))
        probs = self.softmax(z)
        
        # Compute the difference between predicted probabilities and the true one-hot labels
        delta = probs - y_one_hot_encoded
        
        # Compute the gradient (delta.T * X_aug) and average over the size of batch
        grad = np.dot(np.transpose(delta), X_aug) / len(y_one_hot_encoded)
        
        return grad


def kmeans(examples, K, maxIters):
    """
    Perform K-means clustering on |examples|, where each example is a sparse feature vector.

    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    """

    # All unique features from all examples
    all_features = set() # Use set for unique elements
    for example in examples:
        all_features.update(example.keys())
    
    # Order all features indexes, so it's consistent.
    all_features = sorted(all_features)
    # Feature to index dictionary
    feature_index = {}
    for idx, feature in enumerate(all_features):
        feature_index[feature] = idx
    
    # Convert sparse feature vector to numpy array
    np_examples = []
    for ex in examples:
        # Create a zero vector of length all_features
        np_vector = np.zeros(len(all_features))
        
        # Iterate over example dictionary fill the corresponding indices in the numpy array
        for feature, value in ex.items():
            # fill indices in the numpy array
            np_vector[feature_index[feature]] = value 
        
        # Append the resulting dense numpy vector to np_examples
        np_examples.append(np_vector)
    
    # Number of examples
    N = len(np_examples)

    # randomly selecting K examples and initialize the centroids 
    centroids = random.sample(np_examples, K)

    # clusters store index of the nearest centroid
    clusters = [-1] * N  

    for iteration in range(maxIters):
        # Expectation step: assign each example to the nearest centroid
        new_clusters = []
        for example in np_examples:
            # Calculate the distance from example to each centroid. We use second norm.
            distances = []
            for centroid in centroids:
                distance = np.linalg.norm(example - centroid)
                distances.append(distance)
            # Assign the example to the nearest centroid
            new_clusters.append(np.argmin(distances))
        
        # Maximization: compute centroids based on new clusters
        new_centroids = []
        for k in range(K):
            # Get all examples assigned to cluster k
            cluster_examples = []
            for i in range(N):
                if new_clusters[i] == k:
                    cluster_examples.append(np_examples[i])
            if cluster_examples:
                # Compute new centroid = mean of all examples in this cluster
                new_centroids.append(np.mean(cluster_examples, axis=0))
            else:
                # No examples are assigned to this centroid, use the old centroid
                new_centroids.append(centroids[k])
        
        # Check if cluster have changed. No change means function converged.
        if new_clusters == clusters:
            break
        
        # Update centroids and clusters
        centroids = new_centroids
        clusters = new_clusters
    
    # convert numpy back to dictionary for testing.
    dict_centroids = []
    for centroid in centroids:
        sparse_dict = {}
        for idx, value in enumerate(centroid):
            if value != 0:  
                sparse_dict[all_features[idx]] = value
        dict_centroids.append(sparse_dict)

    # Final reconstruction loss squared error
    loss = 0
    for i in range(N):
        loss += np.linalg.norm(np_examples[i] - centroids[clusters[i]]) ** 2

    return dict_centroids, clusters, loss