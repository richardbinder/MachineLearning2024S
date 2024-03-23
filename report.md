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

The Obesity data set has a total of 2111 instances and 16 non-target attributes. The attributes are of type nominal, ordinal or ratio. None of the instances have missing values

### Obesity Attributes

| Attribute                               | Type    | Range                                                |
|-----------------------------------------|---------|------------------------------------------------------|
| Gender                                  | Nominal | Male, Female                                         |
| Age                                     | Ratio   | Number                                               |
| Height                                  | Ratio   | Number in meters                                     |
| Weight                                  | Ratio   | Number in kg                                         |
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