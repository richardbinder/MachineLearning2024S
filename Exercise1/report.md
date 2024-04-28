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

The Obesity data set, contains data from a survey from Mexico, Peru and Colombia in order 
to determine the obesity level of the participants based on their eating habits, 
physical activities, etc. The data set contains `16 non-target attributes` and `1 class attribute`. 
The attributes are of type nominal, ordinal and ratio. None of the instances have missing values.
For us this dataset is interesting to use in a "classification" context, since we can predict 
the obesity level of a person based on their eating habits, physical activities, etc.

- Instances:
    - Total: 2111 instances
    - Training: 75% of the dataset
    - Test: 25% of the dataset
- Attributes:
    - class:
        - `Description`: Obesity level
        - `Type`: ordinal
    - Attributes 2-16:
        - `Description`: 16 attributes describing several characteristics and health aspects 
            of each surveyed person, e.g. weight, age, height, age, smoking habits, drinking habits, etc.
        - `Types`: nominal, ordinal, ratio

### Preprocessing
The following preprocessing steps will be performed.

- We transform ordinal attributes by ordinal encoding, i.e. each category is a number 
such that the original ordering is preserved. Some nominal attributes were already encoded like
this in the dataset, others had to be encoded manually. This applies to the following attributes.
- We use simple label encoding for each remaining nominal attribute (all of them are binary).
- The ratio attributes are left as is for now. 

We then define several preprocessing variations:
- `simple`: Preprocessing as defined above.
- `scaled`: Preprocessing applies min-max scaling to every attribute in the dataset such that 
0 is the minimum and 1 is the maximum.
- `1p`: Only the most important attributes in the dataset are selected for training,
leaving out any attribute with less than 1% importance score.
The score is based on a RandomForestClassifier.
- `1p_scaled`: Same as 1p, but the attributes are also scaled.
- `5p`: Same as 1p, but leaving out any attribute with less than 5% importance.
- `5p_scaled`: Same as 5p. but the attributes are also scaled.


### KNN (K Nearest Neighbours Classifier)

#### Exploring Classification Configurations

We explore KNN parameters (k, leaf size, weight) on the `scaled` dataset. 
Only the choice of k seems to have any impact on the results.
The best k appears to be 1, although 5 ist not far off.
Trivially, too high values of k have a negative impact on all metrics 
due to "blurring" the classes together.  

|   k |   leaf_size | weight   |   Accuracy |   Precision |   Recall |       F1 |
|----:|------------:|:---------|-----------:|------------:|---------:|---------:|
|   1 |           1 | uniform  |   0.780303 |    0.769977 | 0.777837 | 0.774867 |
|   1 |           5 | uniform  |   0.780303 |    0.769977 | 0.777837 | 0.774867 |
|   1 |          20 | uniform  |   0.780303 |    0.769977 | 0.777837 | 0.774867 |
|   1 |           1 | distance |   0.780303 |    0.769977 | 0.777837 | 0.774867 |
|   1 |           5 | distance |   0.780303 |    0.769977 | 0.777837 | 0.774867 |
|   1 |          20 | distance |   0.780303 |    0.769977 | 0.777837 | 0.774867 |
|   5 |           1 | uniform  |   0.75947  |    0.748841 | 0.757308 | 0.751506 |
|   5 |           5 | uniform  |   0.75947  |    0.748841 | 0.757308 | 0.751506 |
|   5 |          20 | uniform  |   0.75947  |    0.748841 | 0.757308 | 0.751506 |
|   5 |           1 | distance |   0.770833 |    0.760931 | 0.768478 | 0.760012 |
|   5 |           5 | distance |   0.770833 |    0.760931 | 0.768478 | 0.760012 |
|   5 |          20 | distance |   0.770833 |    0.760931 | 0.768478 | 0.760012 |
|  20 |           1 | uniform  |   0.683712 |    0.66395  | 0.681697 | 0.662868 |
|  20 |           5 | uniform  |   0.683712 |    0.66395  | 0.681697 | 0.662868 |
|  20 |          20 | uniform  |   0.683712 |    0.66395  | 0.681697 | 0.662868 |
|  20 |           1 | distance |   0.714015 |    0.701612 | 0.712415 | 0.695915 |
|  20 |           5 | distance |   0.714015 |    0.701612 | 0.712415 | 0.695915 |
|  20 |          20 | distance |   0.714015 |    0.701612 | 0.712415 | 0.695915 |
| 100 |           1 | uniform  |   0.549242 |    0.559037 | 0.546023 | 0.496239 |
| 100 |           5 | uniform  |   0.549242 |    0.559037 | 0.546023 | 0.496239 |
| 100 |          20 | uniform  |   0.549242 |    0.559037 | 0.546023 | 0.496239 |
| 100 |           1 | distance |   0.609848 |    0.627705 | 0.610395 | 0.565435 |
| 100 |           5 | distance |   0.609848 |    0.627705 | 0.610395 | 0.565435 |
| 100 |          20 | distance |   0.609848 |    0.627705 | 0.610395 | 0.565435 |

#### Exploring Preprocessing Variations

Using the best found parameters from above (k=1, leaf_size=5, weight=distance),
we evaluate performance on all preprocessing variations.
Scaling has a negative impact on performance here.
This is a hint at the fact that simple min-max scaling does not make sense for all
attributes.
Some attribute values become too close to each other, making it harder for the KNN model
to separate the classes.
Leaving out attributes with less than 1% importance reduces complexity and has a positive impact.
This is not surprising as KNN tends to perform better with more simple data.
Leaving out attributes with less than 5% importance on the other hand leaves 
out too much information and thus has a negative impact.

| name      |   accuracy |   precision |   recall |   f1_score |
|:----------|-----------:|------------:|---------:|-----------:|
| simple    |   0.88447  |    0.880534 | 0.886004 |   0.876739 |
| scaled    |   0.780303 |    0.769977 | 0.777837 |   0.774867 |
| 1p        |   0.890152 |    0.88684  | 0.892136 |   0.884247 |
| 1p_scaled |   0.820076 |    0.813363 | 0.817979 |   0.816772 |
| 5p        |   0.863636 |    0.861078 | 0.863407 |   0.858544 |
| 5p_scaled |   0.856061 |    0.850314 | 0.854285 |   0.854301 |

### Neural Network

We explore Neural Network parameters (hidden layers, activation function, solving optimization algorithm) 
on the `scaled` dataset. 
Interestingly, the choice of hidden layers and optimization algorithm doesn't seem to matter at 
all as long as the activation function is the identity function 
(which ofc makes the Neural Network a simple linear function).
The problem may predominantly involve linear relationships 
(e.g. higher weight -> proportionally higher chance for obesity).
In such cases, introducing non-linearity (through tanh or ReLU) could decrease performance.
This hypothesis can easily be tested with our Random Forest Classifier,
as it is very good at decoding linearly dependent variables.
On another note, ADAM is the most consistent algorithm in all tests, most likely due to
imbalanced classes, e.g. normal weight is much more likely than Obesity Type III. 
ADAM assigns higher learning rate to underrepresented classes.

#### Exploring Classification Configurations

| hidden_layers   | activation   | solver   |   Accuracy |   Precision |   Recall |        F1 |
|:----------------|:-------------|:---------|-----------:|------------:|---------:|----------:|
| (5, 3)          | relu         | adam     |   0.967803 |   0.967364  | 0.965452 | 0.967641  |
| (5, 3)          | relu         | lbfgs    |   0.17803  |   0.0254329 | 0.142857 | 0.0538098 |
| (5, 3)          | relu         | sgd      |   0.636364 |   0.638873  | 0.633966 | 0.598734  |
| (5, 3)          | identity     | adam     |   0.971591 |   0.970786  | 0.970211 | 0.971557  |
| (5, 3)          | identity     | lbfgs    |   0.960227 |   0.958614  | 0.960267 | 0.960043  |
| (5, 3)          | identity     | sgd      |   0.960227 |   0.960132  | 0.959208 | 0.959994  |
| (5, 3)          | logistic     | adam     |   0.903409 |   0.920436  | 0.903379 | 0.897618  |
| (5, 3)          | logistic     | lbfgs    |   0.954545 |   0.954381  | 0.955411 | 0.954095  |
| (5, 3)          | logistic     | sgd      |   0.185606 |   0.0525127 | 0.151927 | 0.0670128 |
| (5, 3)          | tanh         | adam     |   0.9375   |   0.935945  | 0.938868 | 0.937046  |
| (5, 3)          | tanh         | lbfgs    |   0.950758 |   0.9508    | 0.951346 | 0.950301  |
| (5, 3)          | tanh         | sgd      |   0.787879 |   0.794701  | 0.791967 | 0.780513  |
| (20, 10, 5)     | relu         | adam     |   0.844697 |   0.882456  | 0.864548 | 0.83549   |
| (20, 10, 5)     | relu         | lbfgs    |   0.844697 |   0.884402  | 0.865057 | 0.833491  |
| (20, 10, 5)     | relu         | sgd      |   0.149621 |   0.0213745 | 0.142857 | 0.0389459 |
| (20, 10, 5)     | identity     | adam     |   0.975379 |   0.974107  | 0.974472 | 0.975323  |
| (20, 10, 5)     | identity     | lbfgs    |   0.969697 |   0.967337  | 0.969088 | 0.96948   |
| (20, 10, 5)     | identity     | sgd      |   0.967803 |   0.967     | 0.966913 | 0.967699  |
| (20, 10, 5)     | logistic     | adam     |   0.943182 |   0.943696  | 0.9473   | 0.942971  |
| (20, 10, 5)     | logistic     | lbfgs    |   0.939394 |   0.939337  | 0.937069 | 0.939342  |
| (20, 10, 5)     | logistic     | sgd      |   0.149621 |   0.0213745 | 0.142857 | 0.0389459 |
| (20, 10, 5)     | tanh         | adam     |   0.952652 |   0.951411  | 0.952305 | 0.952316  |
| (20, 10, 5)     | tanh         | lbfgs    |   0.954545 |   0.952961  | 0.95293  | 0.954297  |
| (20, 10, 5)     | tanh         | sgd      |   0.893939 |   0.89433   | 0.894488 | 0.89179   |

#### Exploring Preprocessing Variations

Using the best found parameters from above (hidden layers=(5,3), activation=identity, 
solver=adam), we evaluate performance on all preprocessing variations.
Contrary to KNN, scaling has a very positive impact on performance in our Neural Network.
This makes sense, because attributes in an NN are summed up, and therefore need to be 
of similar magnitude to be weighted equally by the model. 

| name      |   accuracy |   precision |   recall |   f1_score |
|:----------|-----------:|------------:|---------:|-----------:|
| simple    |   0.850379 |    0.844154 | 0.845462 |   0.848653 |
| scaled    |   0.971591 |    0.970786 | 0.970211 |   0.971557 |
| 1p        |   0.850379 |    0.845158 | 0.841707 |   0.849065 |
| 1p_scaled |   0.964015 |    0.963361 | 0.962814 |   0.963803 |
| 5p        |   0.672348 |    0.660342 | 0.66907  |   0.664908 |
| 5p_scaled |   0.700758 |    0.694385 | 0.697014 |   0.698605 |


### Random Forest

#### Exploring Classification Configurations

We explore Random Forest parameters (leaf nodes, maximum depth, maximum features) 
on the `scaled` dataset. 
Finding the best parameters here is a simple matter of brute forcing.
Choosing too few leaf nodes or too little depth underfits the dataset.
Too many features (splits per node) overfits the dataset.
The fact that the performance of the Random Forest is very similar to the Neural Network
is another hint that the variables are mostly linearly dependent on each other.

|   leaf_nodes |   max_depth |   max_features |   Accuracy |   Precision |   Recall |       F1 |
|-------------:|------------:|---------------:|-----------:|------------:|---------:|---------:|
|            5 |           5 |              5 |   0.695076 |    0.677258 | 0.697882 | 0.67563  |
|            5 |           5 |             20 |   0.600379 |    0.540717 | 0.62991  | 0.512041 |
|            5 |          20 |              5 |   0.695076 |    0.677258 | 0.697882 | 0.67563  |
|            5 |          20 |             20 |   0.600379 |    0.540717 | 0.62991  | 0.512041 |
|            5 |        1000 |              5 |   0.695076 |    0.677258 | 0.697882 | 0.67563  |
|            5 |        1000 |             20 |   0.600379 |    0.540717 | 0.62991  | 0.512041 |
|          100 |           5 |              5 |   0.850379 |    0.84495  | 0.849312 | 0.849458 |
|          100 |           5 |             20 |   0.831439 |    0.838974 | 0.828217 | 0.831849 |
|          100 |          20 |              5 |   0.935606 |    0.936145 | 0.934806 | 0.936134 |
|          100 |          20 |             20 |   0.94697  |    0.947067 | 0.948022 | 0.946945 |
|          100 |        1000 |              5 |   0.935606 |    0.936145 | 0.934806 | 0.936134 |
|          100 |        1000 |             20 |   0.94697  |    0.947067 | 0.948022 | 0.946945 |
|         5000 |           5 |              5 |   0.850379 |    0.84495  | 0.849312 | 0.849458 |
|         5000 |           5 |             20 |   0.831439 |    0.838974 | 0.828217 | 0.831849 |
|         5000 |          20 |              5 |   0.945076 |    0.94558  | 0.945526 | 0.945411 |
|         5000 |          20 |             20 |   0.94697  |    0.947067 | 0.948022 | 0.946945 |
|         5000 |        1000 |              5 |   0.945076 |    0.94558  | 0.945526 | 0.945411 |
|         5000 |        1000 |             20 |   0.94697  |    0.947067 | 0.948022 | 0.946945 |

#### Exploring Preprocessing Variations

Using the best found parameters from above (leaf nodes=100, max_depth=20, max_features=5), 
we evaluate performance on all preprocessing variations.
Scaling has no impact at all here. 
This makes sense, as decision trees don't really care about the magnitude of variables.
If the values are 100 times larger, then the decision tree boundaries are automatically
scaled by 100 as well.
Just like with KNN, the simplest data preprocessing that still contains
enough information (1p) has the best performance.

| name      |   accuracy |   precision |   recall |   f1_score |
|:----------|-----------:|------------:|---------:|-----------:|
| simple    |   0.935606 |    0.936145 | 0.934806 |   0.936134 |
| scaled    |   0.935606 |    0.936145 | 0.934806 |   0.936134 |
| 1p        |   0.954545 |    0.954663 | 0.954709 |   0.954635 |
| 1p_scaled |   0.952652 |    0.952832 | 0.952697 |   0.952773 |
| 5p        |   0.871212 |    0.867758 | 0.870318 |   0.870439 |
| 5p_scaled |   0.871212 |    0.867758 | 0.870318 |   0.870439 |














# Dataset #4 - 
