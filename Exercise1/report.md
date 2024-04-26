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

In this test we test both scaled data

#### KNN with feature selection (importance > 0.05)
| scaling | k             | weights | Accuracy | Precision | Recall | F1  |
|---------|---------------|---------|----------|-----------|--------|-----|
| False   | 10            | uniform | 0.916     | 0.94      | 0.90   | 0.914|
| False   | 5 ("Optimal") | uniform | 0.93     | 0.94      | 0.92   | 0.929|
| True    | 10            | uniform | 0.944     | 0.95      | 0.94   | 0.944|
| True    | 18 ("Optimal") | uniform | 0.972   | 0.98      | 0.97   | 0.972|


#### KNN with feature selection (importance > 0.1)
| scaling | k             | weights | Accuracy | Precision | Recall | F1  |
|---------|---------------|---------|----------|-----------|--------|-----|
| False   | 10            | uniform | 0.88     | 0.91      | 0.87   | 0.886|
| False   | 9 ("Optimal") | uniform | 0.902    | 0.91      | 0.89   | 0.901|
| True    | 10            | uniform | 0.94     | 0.95      | 0.94   | 0.944|
| True    | 3 ("Optimal") | uniform | 0.94     | 0.95      | 0.94   | 0.944|


#### KNN with all features
| scaling | k             | weights | Accuracy | Precision | Recall | F1  |
|---------|---------------|---------|----------|-----------|--------|-----|
| True    | 19 ("Optimal") | distance| 0.972   | 0.98      | 0.97   | 0.972|


### Neural Networks
### Random Forest

## Conclusions

# Dataset #2 - Loan
# Dataset #3 - Estimation of Obesity Levels 
# Dataset #4 - 
