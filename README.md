# Power-system-load-prediction

In this project, I use the load and temperature data recorded by a set of utility companies in the US. There are 20 zones, each with a different pattern of hourly load values. There are 11 temperature stations, each with a different location. I start by looking for patterns or correlations among the temperature stations and the load values in each zone. If there is a strong correlation between a temperature station and the load values in a specific zone, then I use that station's temperature data to predict the load values for that specific zone. 

Before finding a correlation, some preprocessing is done on the dataset to take out unwanted data. The correlation search is implemented using LinearRegression technique and obtaining the correlation scores. 

I went ahead to train two models: One using the Decision Tress Regressor and the other using K-Nearest Neighbor Regressor just to compare their performance on the data. The model with a better score score on the test set (Decision Treee Regressor had a better score) was used to predict the load values for the 20 different zones. The predicted loads are written in a csv file. 
