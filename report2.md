# Machine Learning, Exercise 0 Report
#### Nicolas Bernal (12347489), Richard Binder (01425185), Victor Olusesi (11776826)


# Data Set Choice
For this exercise we chose the data sets [Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) 
and [Obesity](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition).
They have significant differences (diversity requirement), both have
meaningful underlying problems, and enough data for Machine Learning to be useful.
The attributes of both datasets have sufficient descriptions in the original upload page.
The differences between the datasets are showcased in the table below.

| Obesity                                                 | Bike Sharing                                                    |
|---------------------------------------------------------|-----------------------------------------------------------------|
| ~2k instances                                           | ~17k instances                                                  |
| Classification problem                                  | Regression problem                                              |
| Mostly categorical attributes                           | Mostly numerical and cyclical attributes (e.g. hour of the day) |
| No cyclical attributes                                  | Most ordinal attributes are cyclical                            |
| Numerical attributes only require min-max-normalization | Numerical attributes require standardization (e.g. count)       |

## Obesity
The Obesity data set, contains data from a survey of `2111 people (instances)` from Mexico, 
Peru and Colombia in order to determine the obesity level of the participants based on their eating habits, 
physical activities, etc. The data set contains `16 non-target attributes` and `1 class attribute`. 
The attributes are of type nominal, ordinal and ratio. None of the instances have missing values.

For us this dataset is interesting to use in a "classification" context, since we can predict the obesity level 
of a person based on their eating habits, physical activities, etc.

### Obesity Attributes

| Attribute                                        | Type    | Range                                                |
|--------------------------------------------------|---------|------------------------------------------------------|
| Gender                                           | Nominal | Male, Female                                         |
| Age                                              | Ratio   | Number (from 14 to 61)                               |
| Height                                           | Ratio   | Number in meters (from 1.45 to 1.98)                 |
| Weight                                           | Ratio   | Number in kg (from 39 to 173)                        |
| Family_history_with_overweight                   | Nominal | Yes or No                                            |
| FAVC (Frequently high caloric food)              | Nominal | Yes or No                                            |
| FCVC (Vegetables in your meals)                  | Ordinal | Never, Sometimes or Always                           |
| NCP (Amount of daily main meals)                 | Ordinal | 1-2, 3, >3                                           |
| CAEC (Eating food between meals)                 | Ordinal | No, Sometimes, Frequently, Always                    |
| SMOKE                                            | Nominal | Yes or No                                            |
| CH2O (Daily water intake)                        | Ordinal | <1L, 1-2L, >2L                                       |
| SCC (Monitoring of daily calory intake)          | Nominal | Yes or No                                            |
| FAF (Days a week with physical activity)         | Ordinal | None, 1-2 Days, 2-4 days, 4-5 days                   |
| TUE (Daily time spent using electronic devices)  | Ordinal | 0-2 hours, 3-5 hours, >5 hours                       |
| CALC (Alcohol consumption)                       | Ordinal | No drinking, sometimes, frequently, always           |
| MTRANS (usually used transportation)             | Nominal | Car, Motorbike, Bike, Public transportation, Walking |

### Obesity Target Attribute

| Attribute                  | Type    | Range                                                                                                                              |
|----------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------|
| NObeyesdad (Obesity level) | Ordinal | Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, and Obesity Type III |

### Preprocessing
The following preprocessing steps will be performed.

- We transform ordinal attributes by ordinal encoding, i.e. each category is a number such that the original
ordering is preserved. 
This applies to the following attributes:
    - `FCVC` (Vegetables in your meals)
    - `NCP` (Amount of daily main meals)
    - `CH2O` (Daily water intake)
    - `FAF` (Days a week with physical activity)
    - `TUE` (Daily time spent using electronic devices)
    - `CAEC` (Eating food between meals)
    - `CALC` (Alcohol consumption)
    - `NObeyesdad` (Obesity level)

- We use label encoding for each nominal attribute with only 2 categories, i.e. the first category is assigned 0,
the second is assigned 1.
  - `Gender` (Male = 1, Female = 0)
  - `Family_history_with_overweight`
  - `SMOKE`
  - `SCC` (Monitoring of daily calory intake)
  - `FAVC` (Frequently high caloric food)

- We use one-hot encoding for each nominal attribute, i.e. each category in a nominal attribute becomes a binary attribute.
Ordinal encoding does not make sense here since the categories don't have any underlying order.
  - `MTRANS` (usually used transportation)

- We normalize ratio attributes to a range of 0 to 1. This is done to help the model 
to keep all values in a similar range, helping the ML algorithm to understand the data better without losing information. 
Range encoding makes sense here since humans never exceed a certain threshold (130 years old, 2.5m tall, weight 300kg). 
    - `Age`
    - `Height`
    - `Weight`


### Important Attributes
- `NObeyesdad` (Obesity level) is the class for the Obesity dataset. 
It is an ordinal attribute with 7 different classes. 
This attribute is important because it is the attribute we want to predict.

- `Weight` and `Height` are important attributes because they are the main factors that determine the obesity level of a person. 
Also the `Gender` attribute can influence the obesity level (BMI) for kids and teenagers between 2 and 20 years old.

- `Family_history_with_overweight`, `FAVC`, `SMOKE`, `CALC`, `MTRANS`, `NCP`, `FAF` and `FCVC` are factors 
that can influence directly or indirectly the obesity level of a person. 
For example, a person that has a family history with overweight is more likely to be overweight too. 
Or a person that has a high caloric food consumption is more likely to be overweight. 
These can also be interesting attributes to be predicted since the obesity level can be calculated based on 
`Weight` and `Height` therefore instead of predicting the obesity level, 
we can predict some of these attributes to find the main reasons that lead to a certain obesity level. 


### Histogram of the Obesity Dataset
![Histogram of the Obesity Dataset](./histograms/obesity_histogram.png)

## Bike Sharing
The Bike Sharing data set contains renting data from a bike sharing system in Washington D.C. from 2011 to 2012. 
The data set contains `17379 instances` and `16 non-target attributes`. 
The attributes are of type nominal, ordinal, interval and ratio. 
One additional challenge in this dataset is that some ordinal attributes are cyclical (e.g. hour of the day).
None of the instances have missing values. 
This dataset aims to understand how different factors affect the number of bike rentals in a bike sharing system, 
including weather conditions, time of the day, holidays, etc. 

For us this dataset is interesting to use in a "regression" context, 
since we can predict the number of bike rentals based on the weather conditions, time of the day, holidays, etc.

### Bike Sharing Attributes

| Attribute   | Type     | Range                                                                                   |
|-------------|----------|-----------------------------------------------------------------------------------------|
| instant     | Ratio    | Unique ID for records                                                                   |
| dteday      | Interval | Dates from 2011 to 2012                                                                 |
| season      | Nominal  | 1: winter, 2: spring, 3: summer, 4: fall                                                |
| yr          | Nominal  | 0: 2011, 1: 2012                                                                        |
| mnth        | Ordinal  | 1 to 12                                                                                 |
| hr          | Ordinal  | 0 to 23                                                                                 |
| holiday     | Nominal  | 0: no, 1: yes                                                                           |
| weekday     | Ordinal  | 0 to 6                                                                                  |
| workingday  | Nominal  | 0: no, 1: yes                                                                           |
| weathersit  | Nominal  | 1 to 4 (various weather conditions)                                                     |
| temp        | Ratio    | Normalized temperature in Celsius (0.02 to 1). The values are divided to 41 (max)       |
| atemp       | Ratio    | Normalized feeling temperature in Celsius  (0 to 1). The values are divided to 50 (max) |
| hum         | Ratio    | Normalized humidity (0 to 1). The values are divided to 100 (max).                      |
| windspeed   | Ratio    | Normalized wind speed       (0 to 0.85). The values are divided to 67 (max)             |
| casual      | Ratio    | Count of casual users       (0 to 367)                                                  |
| registered  | Ratio    | Count of registered users   (0 to 886)                                                  |
| cnt         | Ratio    | Count of total rental bikes (1 to 977)                                                  |


### Bike Sharing Target Attribute

| Attribute  | Type     | Range                       |
|------------|----------|-----------------------------|
| cnt        | Ratio    | Count of total rental bikes |

### Preprocessing
We will perform the following changes to preprocess the data:
- Remove the `instant` attribute, since it is just an ID for the records and does not provide any useful information.
- Remove the `dteday` attribute, since we already have the `yr`, `mnth`, `hr`, `holiday`, `weekday`, and `workingday` attributes, 
which provide the same information in a more useful format.
- Transform each cyclical attribute into two attributes, of which the first contains the cosine value,
and the second contains the sine value of the original attribute. As an example, the first and last hour of the day 
(0 and 24) are then the same (cosine=1, sine=0), with a smooth transition between the two. This applies to the following attributes:
  - `yr`
  - `season`
  - `mnth`
  - `weekday`
  - `hr`
- Normalize the count attributes of the dataset to have mean=0 and standard deviation=1.
  - `casual`
  - `registered`
  - `cnt` 
- Use label encoding for attributes with only two categories, i.e. the first category is 0, the second is 1.
  - `holiday`
  - `workingday`
- Use one-hot encoding for the remaining nominal attributes with more than two categories.
  - `weathersit`


### Important Attributes
- `cnt` is the target attribute for the Bike Sharing dataset. 
It is a ratio attribute that represents the total number of bikes rented in a day. 
This attribute is important because it is the attribute we want to predict.
- `temp`, `atemp`, `hum`, and `windspeed` are the weather attributes that may influence the number of bike rentals.
For example, the temperature and humidity can influence the number of bike rentals, 
since people are more likely to rent bikes when the weather is nice. 
The wind speed can also influence the number of bike rentals, since people are less likely to rent bikes when it is windy.
- `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, and `workingday` are the time attributes 
that may influence the number of bike rentals, since depending on the season, year, month, hour, holiday, weekday, 
and working day, the number of bike rentals can vary. For example, people are more likely to rent bikes in the summer 
than in the winter, or on a working day than on a holiday.

### Histogram of the Bike Sharing Dataset
![Histogram of the Bike Sharing Dataset](./histograms/bike_renting_histograms.png)