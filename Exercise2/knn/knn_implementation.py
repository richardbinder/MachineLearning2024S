import numpy as np
from sklearn.metrics.pairwise import (
    euclidean_distances,
    cosine_similarity,
    manhattan_distances
)

class KNN:
    def __init__(self, k=5, distance='euclidean', weighted=False):
        """
        When initializing the class, we can specify the number of neighbors to consider (default is 5), 
        and the distance metric to use (default is 'euclidean').
        """

        # Assign the number of neighbors to consider
        self.k = k
        
        # Assign the distance metric to use
        self.distance = distance
        
        # Assign the weighted parameter, defines if the KNN is weighted or not
        self.weighted = weighted
        
    
    def fit(self, x, y):
        """
        The fit method saves the training data, separating the features and the target variable.
        For KNN, there is not actual training (adjusting parameters) just saving the data.
        """

        # Save the training data in the variables train_x
        self.train_x = x
        
        # Save the training target in the variable train_y
        self.train_y = y


    def predict(self, x):
        """
        Perform regression on samples in x. 
        
        Returns: a list of predictions for each array/record in the test set (x).
        """
        
        # Assign a distance function from sklearn on the variable `distance_function` based on the distance metric specified
        if self.distance == 'cosine':
            distance_function = cosine_similarity
        elif self.distance == 'manhattan':
            distance_function = manhattan_distances
        else:
            distance_function = euclidean_distances
        
        
        # List to store the predictions as a tuple 
        predictions = []

        # Calculate distances between all test points and all training points, using the function specified in `distance_function`
        # Returns a matrix of distances between all test points and all training points
        distances = distance_function(x, self.train_x)
        
        # Invert cosine similarities, since for this one higher values mean closer points. 
        # For consistency, we want lower values to mean closer points.
        if self.distance == 'cosine':
            distances = 1 - distances

        # For each array/record in the test set (x), get the average of the k-nearest neighbors
        for i in range(len(x)):
            
            # Process the distances list to get the predicted target value
            prediction = self._process_closest_neighbors(distances[i])

            # Compute the mean of the target values of the k-nearest neighbors
            predictions.append(prediction)
        
        return predictions
    

    def _process_closest_neighbors(self, distances):
        """
        - Convert the distances to a list of tuples with the distance and the target value
        - Sort the distances list by the lowest distance first
        - Filter the k-nearest neighbors from the distances list
        - Get the target value of the k-nearest neighbors
        
        Returns: the mean of the target values of the k-nearest neighbors.
        """

        # Create a list of tuples with the distance and the target value
        dist_with_targets = list(zip(distances, self.train_y))
        
        # Sort the distances list by the lowest distance first
        dist_with_targets.sort(key=lambda x: x[0])

        # Filter the k-nearest neighbors from the distances list
        neighbors = dist_with_targets[:self.k]

        # For each neighbor in neighbors, get the target value
        neighbors_targets = [neighbor[1] for neighbor in neighbors]
        
        # ---- This is the part that changes between KNN and Weighted KNN, this is just KNN ----
        if not self.weighted:
            return np.mean(neighbors_targets)
        
        # ---- This is the part that changes between KNN and Weighted KNN, this is Weighted KNN ----
        # For each neighbor in neighbors, get the distance value
        neighbors_distances = [neighbor[0] for neighbor in neighbors]
        
        # Invert distances to use them as weights (closer points get higher weights)
        # Add a small value to the distance (in this case 1e-5) to avoid division by zero
        weights = [1 / (dist + 1e-5) for dist in neighbors_distances]

        # Calculate the weighted average
        # np.dot is the dot product the weights and the target values
        # np.sum is the sum of the weights
        weighted_average = np.dot(neighbors_targets, weights) / np.sum(weights)
        
        return weighted_average
