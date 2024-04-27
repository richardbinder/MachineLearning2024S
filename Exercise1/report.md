# Machine Learning - Exercise 1 Report
## Authors:
- Nicolas Bernal (12347489)
- Richard Binder (01425185)
- Victor Olusesi (11776826)

# Dataset #1 - Breast Cancer

## Exploratory

### Description of the dataset
The "Breast cancer diagnostic" dataset contains 32 attributes and 285 instances. The attributes and instances are described as follows:

- Instances:
    - Training: 213 instances (75% of the dataset)
    - Validation: 72 instances (25% of the dataset)
    - Total: 285 instances
- Attributes:
    - id:
        - `Description`: ID of the person.
        - `Type`: integer
    - class:
        - `Description`: Diagnosis of the person, `True` if the person has malignant a tumor, `False` otherwise.
        - `Type`: boolean
    - Attributes 3-32:
        - `Description`: 29 attributes describing the shape, texture, area, etc. of the tumor in the breast; in order to determine whether the tumor is malignant or benign. 
        - `Type`: double

### Visualization of the dataset
In the following histograms we can see the distribution of the attributes in the dataset (on the right) and the distribution of the attributes for the two classes (on the left).

The histograms show that some attributes have a clear separation between the two classes, while others have a significant overlap. This might suggest that the attributes with the highest separation in their distributions might be the ones that are most relevant for the classification, since we can see that the values of these attributes are more likely to be different for the two classes.

### Histograms of the attributes with the `highest` separation between the two classes
![Histograms01](/Exercise1/BreastCancer/plots/interesting_attributes.png)

### Histograms of the attributes with the `lowest` separation between the two classes
![Histograms02](/Exercise1/BreastCancer/plots/non_so_interesting_attributes.png)

## Preprocessing of the data
- `Missing values`: There are no missing values in the dataset.
- `Scaling`: We will scale the data using the `StandardScaler` from `sklearn.preprocessing`. In order to make the data more suitable for the models, the data will be scaled to have a mean of 0 and a standard deviation of 1.
- `Encoding`: The class attribute will be encoded to 0 and 1, where 0 represents a benign tumor and 1 represents a malignant tumor.
- `Attribute selection`: 
    - We will drop the `id` attribute, as it is irrelevant for the classification.
    - We will use a `RandomForestClassifier` to select the most important features, in order to make some test with a reduced number of features.

## Comparison of different classifiers
### KNN

For the KNN classifier, we made several test with different variations of the hyperparameters (such as the number of neighbors and the distance metric), and also with different ways of preprocessing the data (such as scaling and feature selection). 

To obtain the optimal K for each test, we used cross-validation varying number of folds for each case, and we selected the K with the highest accuracy, doing this proccess with values of K from 1 to 100.

Also, for feature selection, we used a `RandomForestClassifier` to select the most important features, and we made tests with the selected features and with all the features. In this case we decided to pick those features that have an importance greater than 0.05 for one group of tests, and greater than 0.1 for another group of tests.

The selected features are, that have an importance greater than 0.05 are:
- concavePointsWorst
- areaWorst
- concavePointsMean
- radiusWorst
- concavityMean
- perimeterMean
- perimeterWorst
- radiusMean
- areaMean

The selected features are, that have an importance greater than 0.1 are:
- concavePointsWorst
- areaWorst
- concavePointsMean

The results of the tests are shown in the following tables:

#### KNN with all features
| scaling | k             | weights | Accuracy | Precision | Recall | F1  |
|---------|---------------|---------|----------|-----------|--------|-----|
| False   | 10            | uniform | 0.916    | 0.94      | 0.90   | 0.914|
| False   | 10            | distance| 0.93     | 0.95      | 0.92   | 0.929|
| False   | 5 ("Optimal") | uniform | 0.93     | 0.94      | 0.92   | 0.929|
| False   | 5 ("Optimal") | distance| 0.94     | 0.95      | 0.94   | 0.944|
| True    | 10            | uniform | 0.94     | 0.96      | 0.93   | 0.943|
| True    | 10            | distance| 0.94     | 0.96      | 0.93   | 0.943|
| True    | 19 ("Optimal") | uniform | 0.972   | 0.98      | 0.97   | 0.972|
| True    | 19 ("Optimal") | distance| 0.972   | 0.98      | 0.97   | 0.972|


#### KNN with feature selection (importance > 0.05)
| scaling | k             | weights | Accuracy | Precision | Recall | F1  |
|---------|---------------|---------|----------|-----------|--------|-----|
| False   | 10            | uniform | 0.916    | 0.94      | 0.90   | 0.914|
| False   | 10            | distance| 0.93     | 0.946     | 0.916  | 0.929|
| False   | 5 ("Optimal") | uniform | 0.93     | 0.94      | 0.92   | 0.929|
| False   | 5 ("Optimal") | distance| 0.944    | 0.948     | 0.938  | 0.944|
| True    | 10            | uniform | 0.944    | 0.95      | 0.94   | 0.944|
| True    | 10            | distance| 0.958    | 0.959     | 0.954  | 0.958|
| True    | 18 ("Optimal")| uniform | 0.972    | 0.98      | 0.97   | 0.972|
| True    | 18 ("Optimal")| distance| 0.972    | 0.972     | 0.966  | 0.972|


#### KNN with feature selection (importance > 0.1)
| scaling | k             | weights | Accuracy | Precision | Recall | F1  |
|---------|---------------|---------|----------|-----------|--------|-----|
| False   | 10            | uniform | 0.88     | 0.906     | 0.87   | 0.886|
| False   | 10            | distance| 0.88     | 0.906     | 0.87   | 0.886|
| False   | 9 ("Optimal") | uniform | 0.902    | 0.907     | 0.89   | 0.901|
| False   | 9 ("Optimal") | distance| 0.88     | 0.906     | 0.87   | 0.886|
| True    | 10            | uniform | 0.94     | 0.95      | 0.94   | 0.944|
| True    | 10            | distance| 0.958    | 0.959     | 0.954  | 0.958|
| True    | 3 ("Optimal") | uniform | 0.94     | 0.95      | 0.94   | 0.944|
| True    | 3 ("Optimal") | distance| 0.94     | 0.948     | 0.938  | 0.944|



### Neural Networks

For neural networks we made several tests with different hyperparameters, such as the number of hidden layers, the activation function, the solver, and the scaling of the data. We also made tests with all the features and with the features selected by the `RandomForestClassifier` with an importance greater than 0.05 and 0.1.


For the first part of test, we tried different combinations of hyperparameters (activation and solver) and we made tests with and without scaling the data. The results of the tests are shown in the following tables:

#### Neural network with all features
##### Neural network with all features and no scaling
| scaling   | hidden_layers   | activation   | solver   |   Accuracy |   Precision |   Recall |       F1 |
|:----------|:----------------|:-------------|:---------|-----------:|------------:|---------:|---------:|
| False     | (5, 3)          | relu         | adam     |   0.819444 |    0.860477 | 0.788095 | 0.809631 |
| False     | (5, 3)          | relu         | lbfgs    |   0.583333 |    0.291667 | 0.5      | 0.429825 |
| False     | (5, 3)          | relu         | sgd      |   0.583333 |    0.291667 | 0.5      | 0.429825 |
| False     | (5, 3)          | identity     | adam     |   0.819444 |    0.860477 | 0.788095 | 0.809631 |
| False     | (5, 3)          | identity     | lbfgs    |   0.583333 |    0.291667 | 0.5      | 0.429825 |
| False     | (5, 3)          | identity     | sgd      |   0.916667 |    0.926421 | 0.904762 | 0.915584 |
| False     | (5, 3)          | logistic     | adam     |   0.861111 |    0.903846 | 0.833333 | 0.85461  |
| False     | (5, 3)          | logistic     | lbfgs    |   0.875    |    0.885532 | 0.859524 | 0.872829 |
| False     | (5, 3)          | logistic     | sgd      |   0.583333 |    0.291667 | 0.5      | 0.429825 |
| False     | (5, 3)          | tanh         | adam     |   0.916667 |    0.926421 | 0.904762 | 0.915584 |
| False     | (5, 3)          | tanh         | lbfgs    |   0.583333 |    0.291667 | 0.5      | 0.429825 |
| False     | (5, 3)          | tanh         | sgd      |   0.625    |    0.804348 | 0.55     | 0.517199 |


With the results of the tests, we can see that the best results were obtained with the following hyperparameters:
- Hyperparameters #1:
    - `activation`: identity
    - `solver`: sgd
    - `F1-score`: 0.915584
- Hyperparameters #2:
    - `activation`: tanh
    - `solver`: adam
    - `F1-score`: 0.915584

This results suggest some interesting points:
- For the first configuration, the best results were obtained with the `identity` activation function and the `sgd` solver, shows that the `identity` activation function (which is basically lineal function) showed some synergy with the `sgd` solver, which in (general) work well with non scaled data. 
- In general the `lbfgs` solver showed bad results, since it is a solver that works well with small dimensions, but in this case we are working with 29 features.

##### Neural network with all features and scaling
| scaling   | hidden_layers   | activation   | solver   |   Accuracy |   Precision |   Recall |       F1 |
|:----------|:----------------|:-------------|:---------|-----------:|------------:|---------:|---------:|
| True      | (5, 3)          | relu         | adam     |   0.972222 |    0.977273 | 0.966667 | 0.972066 |
| True      | (5, 3)          | relu         | lbfgs    |   0.958333 |    0.955547 | 0.959524 | 0.958424 |
| True      | (5, 3)          | relu         | sgd      |   0.944444 |    0.956522 | 0.933333 | 0.943723 |
| True      | (5, 3)          | identity     | adam     |   0.972222 |    0.977273 | 0.966667 | 0.972066 |
| True      | (5, 3)          | identity     | lbfgs    |   0.944444 |    0.940625 | 0.947619 | 0.944663 |
| True      | (5, 3)          | identity     | sgd      |   0.972222 |    0.977273 | 0.966667 | 0.972066 |
| True      | (5, 3)          | logistic     | adam     |   0.972222 |    0.977273 | 0.966667 | 0.972066 |
| True      | (5, 3)          | logistic     | lbfgs    |   0.958333 |    0.955547 | 0.959524 | 0.958424 |
| True      | (5, 3)          | logistic     | sgd      |   0.583333 |    0.291667 | 0.5      | 0.429825 |
| True      | (5, 3)          | tanh         | adam     |   0.972222 |    0.977273 | 0.966667 | 0.972066 |
| True      | (5, 3)          | tanh         | lbfgs    |   0.958333 |    0.955547 | 0.959524 | 0.958424 |
| True      | (5, 3)          | tanh         | sgd      |   0.972222 |    0.977273 | 0.966667 | 0.972066 |


With the results of the tests, we can see that the best results were obtained with the following hyperparameters:
- Hyperparameters #1:
    - `activation`: relu
    - `solver`: adam
    - `F1-score`: 0.972066
- Hyperparameters #2:
    - `activation`: identity
    - `solver`: adam & sgd
    - `F1-score`: 0.972066
- Hyperparameters #3:
    - `activation`: tanh
    - `solver`: adam & sgd
    - `F1-score`: 0.972066

In general the results obtained after scaling the data are better than the results obtained without scaling it. This is expected since the neural network works better with scaled data, since the weights are updated in a more efficient way which helps to converge faster to an optimal solution. 

With the results of the tests, we can see that the best result for `F1-score` was ~0.972 with the hyperparameters mentioned before. This results suggest that this is the maximum performance that we can obtain with the current preprocessing and architecture of the neural network.

### Neural network with feature selection

For this part of the test, we took the best hyperparameters and we made tests with the features selected by the `RandomForestClassifier` with an importance greater than 0.05 and 0.1. In this case we decided to only apply the scaling to the data, since it gave the best results in the previous tests. The results of the tests are shown in the following tables:

#### Scaling - feature selection #1 (importance > 0.05)
| scaling   | hidden_layers   | activation   | solver   |   Accuracy |   Precision |   Recall |       F1 |
|:----------|:----------------|:-------------|:---------|-----------:|------------:|---------:|---------:|
| True      | (5, 3)          | relu         | adam     |   0.972222 |    0.977273 | 0.966667 | 0.972066 |
| True      | (5, 3)          | identity     | adam     |   0.958333 |    0.959503 | 0.954762 | 0.958225 |
| True      | (5, 3)          | identity     | sgd      |   0.958333 |    0.959503 | 0.954762 | 0.958225 |
| True      | (5, 3)          | tanh         | adam     |   0.958333 |    0.959503 | 0.954762 | 0.958225 |
| True      | (5, 3)          | tanh         | sgd      |   0.958333 |    0.959503 | 0.954762 | 0.958225 |

#### Scaling - feature selection #2 (importance > 0.1)
| scaling   | hidden_layers   | activation   | solver   |   Accuracy |   Precision |   Recall |       F1 |
|:----------|:----------------|:-------------|:---------|-----------:|------------:|---------:|---------:|
| True      | (5, 3)          | relu         | adam     |   0.583333 |    0.291667 | 0.5      | 0.429825 |
| True      | (5, 3)          | identity     | adam     |   0.944444 |    0.948052 | 0.938095 | 0.944133 |
| True      | (5, 3)          | identity     | sgd      |   0.930556 |    0.937037 | 0.921429 | 0.929925 |
| True      | (5, 3)          | tanh         | adam     |   0.944444 |    0.948052 | 0.938095 | 0.944133 |
| True      | (5, 3)          | tanh         | sgd      |   0.930556 |    0.937037 | 0.921429 | 0.929925 |


#### Comparison of the results of the neural network with feature selection #1 (importance > 0.05) and the neural network with feature selection #2 (importance > 0.1):
![results 0.05 vs 0.1](/Exercise1/BreastCancer/plots/nn_scaled_less_attributes_comparison.png)

Comparing the results of the neural network with feature selection #1 (importance > 0.05) and the neural network with feature selection #2 (importance > 0.1), we can see that the results are a little worse in most cases when we use the features with an importance greater than 0.1 (3 attributes). Also, there is a significant decrease in performance in the fist configuration, where the F1-score is 0.429825, which was the best result obtained in the previous case (importance > 0.05).

The decrease in performance in feature selection #2 (importance > 0.1) is expected, since we are using only 3 features, which are apparently enough to obtain a good performance in the classification task, but we are still losing valuable relations between features that could help to improve the performance of the model.

These results suggest that the features with an importance greater than 0.05 (9 features) are good enough to obtain a good performance in the classification task, since the results are similar to the ones obtained with all the features, and the best results in both cases are the same. This is good because it means that we can reduce the number of features without losing much performance in this task.


### Testing others architectures of the neural network with all features

For the las part of the test, we made tests with different architectures of the neural network, varying the number of hidden layers and the number of neurons in each layer. We used all the features and we applied the scaling to the data since this way of preprocessing gave the best results in the previous tests. The results of the tests are shown in the following tables:

| scaling   | hidden_layers         | activation   | solver   |   Accuracy |   Precision |   Recall |       F1 |
|:----------|:----------------------|:-------------|:---------|-----------:|------------:|---------:|---------:|
| True      | (8, 9)                | relu         | adam     |   0.944444 |    0.956522 | 0.933333 | 0.943723 |
| True      | (8, 9)                | relu         | sgd      |   0.972222 |    0.977273 | 0.966667 | 0.972066 |
| True      | (8, 9)                | identity     | adam     |   0.972222 |    0.977273 | 0.966667 | 0.972066 |
| True      | (8, 9)                | identity     | sgd      |   0.972222 |    0.977273 | 0.966667 | 0.972066 |
| True      | (8, 9)                | tanh         | adam     |   0.972222 |    0.977273 | 0.966667 | 0.972066 |
| True      | (8, 9)                | tanh         | sgd      |   0.972222 |    0.977273 | 0.966667 | 0.972066 |
| True      | (1, 3, 5, 3, 1)       | relu         | adam     |   0.583333 |    0.291667 | 0.5      | 0.429825 |
| True      | (1, 3, 5, 3, 1)       | relu         | sgd      |   0.583333 |    0.291667 | 0.5      | 0.429825 |
| True      | (1, 3, 5, 3, 1)       | identity     | adam     |   0.972222 |    0.977273 | 0.966667 | 0.972066 |
| True      | (1, 3, 5, 3, 1)       | identity     | sgd      |   0.972222 |    0.977273 | 0.966667 | 0.972066 |
| True      | (1, 3, 5, 3, 1)       | tanh         | adam     |   0.583333 |    0.291667 | 0.5      | 0.429825 |
| True      | (1, 3, 5, 3, 1)       | tanh         | sgd      |   0.583333 |    0.291667 | 0.5      | 0.429825 |
| True      | (10, 10, 10)          | relu         | adam     |   0.986111 |    0.988372 | 0.983333 | 0.986075 |
| True      | (10, 10, 10)          | relu         | sgd      |   0.972222 |    0.977273 | 0.966667 | 0.972066 |
| True      | (10, 10, 10)          | identity     | adam     |   0.986111 |    0.988372 | 0.983333 | 0.986075 |
| True      | (10, 10, 10)          | identity     | sgd      |   0.986111 |    0.988372 | 0.983333 | 0.986075 |
| True      | (10, 10, 10)          | tanh         | adam     |   0.986111 |    0.988372 | 0.983333 | 0.986075 |
| True      | (10, 10, 10)          | tanh         | sgd      |   0.972222 |    0.977273 | 0.966667 | 0.972066 |
| True      | (50, 40, 60, 80, 100) | relu         | adam     |   0.972222 |    0.971429 | 0.971429 | 0.972222 |
| True      | (50, 40, 60, 80, 100) | relu         | sgd      |   0.986111 |    0.988372 | 0.983333 | 0.986075 |
| True      | (50, 40, 60, 80, 100) | identity     | adam     |   0.944444 |    0.940625 | 0.947619 | 0.944663 |
| True      | (50, 40, 60, 80, 100) | identity     | sgd      |   0.972222 |    0.971429 | 0.971429 | 0.972222 |
| True      | (50, 40, 60, 80, 100) | tanh         | adam     |   0.944444 |    0.940625 | 0.947619 | 0.944663 |
| True      | (50, 40, 60, 80, 100) | tanh         | sgd      |   0.972222 |    0.971429 | 0.971429 | 0.972222 |

With this tests, we can notice that we got an improvement of the performance of the neural network when we increased the number of neurons in the hidden layers. The best results were obtained with the architecture (10, 10, 10) and in one case with the architecture (50, 40, 60, 80, 100), where the F1-score was 0.986075. This results suggest that the previous architectures were not enough to capture the complexity of the data, and that bigger architectures (more neurons and or more layers) were needed to improve performance in the classification task.

Also, we can see that just increasing the number of neurons in the hidden layers is not enough to improve the performance of the neural network, actually in some cases the performance decreased. We can see in 4/6 cases of the architecture (1, 3, 5, 3, 1) that the performance was significantly worse than most of the previous tests with smaller architectures (3, 5). We have the hypothesis that this might be caused by the layers with less neurons, which might be too small to capture the complexity of the data.

### Random Forest

For the Random Forest classifier model, we made some tests with different hyperparameters, as the maximun leaf nodes, the maximun features and the maximun depth of the trees. As in the previous models, we made tests with all the features and with the features selected by a `RandomForestClassifier` with an importance greater than 0.05 and 0.1.

#### Random Forest with all features
| scaling   | max_leaf_nodes | max_features | max_depth |Accuracy |Precision | Recall|  F1  |
|:----------|:---------------|:-------------|:----------|--------:|---------:|------:|-----:|
| False     | 2              | 2            | 2         |   0.888 | 0.906 | 0.871 |0.886 |
| False     | 10             | 10           | 10        |   0.972 | 0.977 | 0.966 |0.972 |
| False     | 50             | 31           | 31        |   0.972 | 0.977 | 0.966 |0.972 |
| True      | 2              | 2            | 2         |   0.888 | 0.906 | 0.871 |0.886 |
| True      | 10             | 10           | 10        |   0.972 | 0.977 | 0.966 |0.972 |
| True      | 50             | 31           | 31        |   0.972 | 0.977 | 0.966 |0.972 |


#### Random Forest with feature selection (importance > 0.05)
| scaling   | max_leaf_nodes | max_features | max_depth |Accuracy |Precision | Recall|  F1  |
|:----------|:---------------|:-------------|:----------|--------:|---------:|------:|-----:|
| False     | 2              | 2            | 2         |   0.958 | 0.959 | 0.954 |0.958 |
| False     | 10             | 10           | 10        |   0.958 | 0.959 | 0.954 |0.958 |
| False     | 50             | 31           | 31        |   0.958 | 0.959 | 0.954 |0.958 |
| True      | 2              | 2            | 2         |   0.958 | 0.959 | 0.954 |0.958 |
| True      | 10             | 10           | 10        |   0.958 | 0.959 | 0.954 |0.958 |
| True      | 50             | 31           | 31        |   0.958 | 0.959 | 0.954 |0.958 |


#### Random Forest with feature selection (importance > 0.1)
| scaling   | max_leaf_nodes | max_features | max_depth |Accuracy |Precision | Recall|  F1  |
|:----------|:---------------|:-------------|:----------|--------:|---------:|------:|-----:|
| False     | 2              | 2            | 2         |   0.93  | 0.937 | 0.921 |0.929 |
| False     | 10             | 10           | 10        |   0.944 | 0.948 | 0.938 |0.944 |
| False     | 50             | 31           | 31        |   0.944 | 0.948 | 0.938 |0.944 |
| True      | 2              | 2            | 2         |   0.93  | 0.937 | 0.921 |0.929 |
| True      | 10             | 10           | 10        |   0.944 | 0.948 | 0.938 |0.944 |
| True      | 50             | 31           | 31        |   0.944 | 0.948 | 0.938 |0.944 |

## Conclusions

# Dataset #2 - Loan
# Dataset #3 - Estimation of Obesity Levels 
# Dataset #4 - 
