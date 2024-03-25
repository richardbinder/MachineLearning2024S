# Machine Learning, Exercise 0 Report
#### Nicolas Bernal (12347489), Richard Binder (01425185), Victor Olusesi (11776826)


# Data Set Choice
Eucalyptus and Obesity

## Eucalyptus

The Eucalyptus data set has a total of 736 entries and 19 non-target attributes. The attributes are of type nominal, interval or ratio. 95 of the instances have missing values, which totals to 448 missing values.

### Eucalyptus Attributes

| Attribute                                | Type     | Range                                                                                                                                                                                   |
|------------------------------------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Abbrev (abbreviation of   planting site) | Nominal  | Cra,Cly,Nga,Wai,K81,Wak,K82,WSp,K83,Lon,Puk,Paw,K81a,Mor,Wen,WSh                                                                                                                        |
| Rep                                      | Interval | Integer in the range 1-3                                                                                                                                                                |
| Locality                                 | Nominal  | Central_Hawkes_Bay,Northern_Hawkes_Bay,Southern_Hawkes_Bay,Central_Hawkes_Bay_(coastal),   Central_Wairarapa,South_Wairarapa,Southern_Hawkes_Bay_(coastal),Central_Poverty_Bay          |
| Map_Ref                                  | Nominal  | N135_382/137, N116_848/985, N145_874/586, N142_377/957, N158_344/626, N162_081/300, N158_343/625, N151_912/221, N162_097/424, N166_063/197, N146_273/737, N141_295/063, N98_539/567, N151_922/226 |
| Latitute                                 | Nominal | 39__38, 39__00, 40__11, 39__50, 40__57, 41__12, 40__36, 41__08, 41__16, 40__00, 39__43, 82__32                                                                                                  |
| Altitude                                 | Ratio    | Number in meters                                                                                                                                                                        |
| Rainfall                                 | Ratio    | Number in mm/a                                                                                                                                                                          |
| Frosts                                   | Interval | Number in degree celsius                                                                                                                                                                |
| Year (year of planting)                  | Ratio    | Year number                                                                                                                                                                             |
| SP (species code)                        | Nominal  | co, fr, ma, nd, ni, ob, ov, pu, rd, si, mn, ag, bxs, br, el, fa, jo, ka, re, sm, ro, nc, am, cr, pa, ra, te                                                                                                       |
| PMCno (seedlot number)                   | Nominal  | 4-digit number                                                                                                                                                                          |
| DBH (best diameter base height)          | Ratio    | Number in cm                                                                                                                                                                            |
| Hat (Height)                             | Ratio    | Number in meters                                                                                                                                                                        |
| Surv (survival rate)                     | Ratio    | Number in percent                                                                                                                                                                       |
| Vig (Vigour)                             | Interval | Number in the range 1-5                                                                                                                                                                 |
| Ins_res (insect resistance)              | Interval | Number in the range 1-5                                                                                                                                                                 |
| Stem_Fm (stem form)                      | Interval | Number in the range 1-5                                                                                                                                                                 |
| Crown_Fm (crown form)                    | Interval | Number in the range 1-5                                                                                                                                                                 |
| Brnch_Fm (branch form)                   | Interval | Number in the range 1-5                                                                                                                                                                 |

### Eucalyptus Target Attribute

| Attribute | Type    | Range                      |
|-----------|---------|----------------------------|
| Utility   | Ordinal | none,low,average,good,best |

## Obesity
The Obesity data set, contains data from a survey of `2111 people (instances)` from Mexico, Peru and Colombia in order to determine the obesity level of the participants based on their eating habits, physical activities, etc. The data set contains `16 non-target attributes` and `1 class attribute`. The attributes are of type nominal, ordinal and ratio. None of the instances have missing values.


The Obesity data set has a total of **2111 instances** and **16 non-target attributes**. The attributes are of type nominal, ordinal or ratio. None of the instances have missing values

### Obesity Attributes

| Attribute                               | Type    | Range                                                |
|-----------------------------------------|---------|------------------------------------------------------|
| Gender                                  | Nominal | Male, Female                                         |
| Age                                     | Ratio   | Number (from 14 to 61)                               |
| Height                                  | Ratio   | Number in meters (from 1.45 to 1.98)                 |
| Weight                                  | Ratio   | Number in kg (from 39 to 173)                        |
| Family_history_with_overweight          | Nominal | Yes or No                                            |
| FAVC (Frequently high caloric food)     | Nominal | Yes or No                                            |
| FCVC (Vegetables in your meals)         | Ordinal | Never, Sometimes or Always                           |
| NCP (Amount of daily main meals)        | Ordinal | 1-2, 3, >3                                           |
| CAEC (Eating food between meals)        | Ordinal | No, Sometimes, Frequently, Always                    |
| SMOKE                                   | Nominal | Yes or No                                            |
| CH2O (Daily water intake)               | Ordinal | <1L, 1-2L, >2L                                       |
| SCC (Monitoring of daily calory intake) | Nominal | Yes or No                                            |
| FAF (Days a week with physical activity)| Ordinal | None, 1-2 Days, 2-4 days, 4-5 days                   |
| TUE (Daily time spent using electronic devices)| Ordinal | 0-2 hours, 3-5 hours, >5 hours                |
| CALC (Alcohol consumption)              | Ordinal | No drinking, sometimes, frequently, always           |
| MTRANS (usually used transportation)    | Nominal | Car, Motorbike, Bike, Public transportation, Walking |

### Obesity Target Attribute

| Attribute | Type    | Range                      |
|-----------|---------|----------------------------|
| NObeyesdad (Obesity level)   | Ordinal | Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, and Obesity Type III |

### Preprocessing
- We can see that several attributes were distributed into "categories" and then transformed into numerical values. These attributes are:
    - `FCVC` (Vegetables in your meals)
    - `NCP` (Amount of daily main meals)
    - `CH2O` (Daily water intake)
    - `FAF` (Days a week with physical activity)
    - `TUE` (Daily time spent using electronic devices)

    These "categories" doesn't really have strong relationships between them (for example FCVC = 2 (Sometimes) is not "double" than FCVC = 1 ("Never")), therefore these attributes are ordinal and cannot be treated as numerical values. That's why we considered to treat them as nominal attributes, transforming each "category" into a binary attribute.

    Also the attributes:
    - `CAEC` (Eating food between meals)
    - `CALC` (Alcohol consumption)
    - `MTRANS` (usually used transportation) 
    
    Will be also receive the same processing as the ones before due to either be ordinal attributes or being nominal attributes with more than two values.

- For handling nominal attributes easily, we decided to transform the nominal attributes into binary values. For example: Yes = 1, No = 0. This applies to the following attributes:
    - `Gender` (Male = 1, Female = 0)
    - `Family_history_with_overweight`
    - `FAVC` (Frequently high caloric food)
    - `SMOKE`
    - `SCC` (Monitoring of daily calory intake)

- And for the ratio attributes, we decided to normalize them to a range of 0 to 1. This is done to help the model to keep all values in a similar range, helping the ML algorithm to understand the data better without losing information. This applies to the following attributes:
    - `Age`
    - `Height`
    - `Weight`


### Important Attributes
- `NObeyesdad` (Obesity level) is the class for the Obesity dataset. It is an ordinal attribute with 7 different classes. This attribute is important because it is the attribute we want to predict.

- `Weight` and `Height` are important attributes because they are the main factors that determine the obesity level of a person. Also the `Gender` attribute can influence the obesity level (BMI) for kids and teenagers between 2 and 20 years old.

- `Family_history_with_overweight`, `FAVC`, `SMOKE`, `CALC`, `MTRANS`, `NCP`, `FAF` and `FCVC` are factors that can influence directly or indirectly the obesity level of a person. For example, a person that has a family history with overweight is more likely to be overweight too. Or a person that has a high caloric food consumption is more likely to be overweight. These can also be interesting attributes to be predicted since the obesity level can be calculated based on `Weight` and `Height` therefore instead of predicting the obesity level, we can predict some of these attributes to find the main reasons that lead to a certain obesity level. 


### Histogram of the Obesity Dataset
![Histogram of the Obesity Dataset](./histograms/obesity_histogram.png)