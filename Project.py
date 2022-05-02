#Implementing the final project
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import joblib
import sys
from predictions import predictload

#Extracting the data from csv files
temp = pd.read_csv('Temp_history_final.csv')
load = pd.read_csv('Load_history_final.csv')

#Preprocessing.
temp[(temp['year']==2008)&(temp['month']==6)].index
temp=temp.drop(temp[(temp['year']==2008)&(temp['month']==6)].index)
load=load.drop(load[(load['year']==2008)&(load['month']==6)].index)
load['date'] = pd.to_datetime(load[['year', 'month', 'day']])

load['daily_load'] = load.drop(['zone_id','year','month','day'], axis='columns').sum(axis='columns')

monthly_mean_load = load.groupby(['zone_id','month'])['daily_load'].mean().reset_index()

load.rename(columns = { 'h1':'H1',
                        'h2':'H2',
                        'h3':'H3',
                        'h4':'H4',
                        'h5':'H5',
                        'h6':'H6',
                        'h7':'H7',
                        'h8':'H8',
                        'h9':'H9',
                        'h10':'H10',
                        'h11':'H11',
                        'h12':'H12',
                        'h13':'H13',
                        'h14':'H14',
                        'h15':'H15',
                        'h16':'H16',
                        'h17':'H17',
                        'h18':'H18',
                        'h19':'H19',
                        'h20':'H20',
                        'h21':'H21',
                        'h22':'H22',
                        'h23':'H23',
                        'h24':'H24'}, inplace = True)

for i in range (1, 21):
    print('Zone_id', i)
    for j in range(1,12):
        station = temp[temp['station_id']==j]
        zone = load[load['zone_id']==i][['daily_load']]
        regressor = LinearRegression()
        regressor.fit(station, zone)
        score = regressor.score(station, zone)
        #print('Station_Id',j, score)

#Extracting the different stations and zones
station1 = temp[(temp['station_id']==1)]
station1 = station1.reset_index()

station2 = temp[(temp['station_id']==2)]
station2 = station2.reset_index()

station3 = temp[(temp['station_id']==3)]
station3 = station3.reset_index()

station4 = temp[(temp['station_id']==4)]
station4 = station4.reset_index()

station5 = temp[(temp['station_id']==5)]
station5 = station5.reset_index()

station6 = temp[(temp['station_id']==6)]
station6 = station6.reset_index()

station7 = temp[(temp['station_id']==7)]
station7 = station7.reset_index()

station8 = temp[(temp['station_id']==8)]
station8 = station8.reset_index()

station9 = temp[(temp['station_id']==9)]
station9 = station9.reset_index()

station10 = temp[(temp['station_id']==10)]
station10 = station10.reset_index()

zone1 = load[load['zone_id']==1]
zone1 = zone1.reset_index()

zone2 = load[load['zone_id']==2]
zone2 = zone2.reset_index()

zone3 = load[load['zone_id']==3]
zone3 = zone3.reset_index()

zone4 = load[load['zone_id']==4]
zone4 = zone4.reset_index()

zone5 = load[load['zone_id']==5]
zone5 = zone5.reset_index()

zone6 = load[load['zone_id']==6]
zone6 = zone6.reset_index()

zone7 = load[load['zone_id']==7]
zone7 = zone7.reset_index()

zone8 = load[load['zone_id']==8]
zone8 = zone8.reset_index()

zone9 = load[load['zone_id']==9]
zone9 = zone9.reset_index()

zone10 = load[load['zone_id']==10]
zone10 = zone10.reset_index()

zone11 = load[load['zone_id']==11]
zone11 = zone11.reset_index()

zone12 = load[load['zone_id']==12]
zone12 = zone12.reset_index()

zone13 = load[load['zone_id']==13]
zone13 = zone13.reset_index()

zone14 = load[load['zone_id']==14]
zone14 = zone14.reset_index()

zone15 = load[load['zone_id']==15]
zone15 = zone15.reset_index()

zone16 = load[load['zone_id']==16]
zone16 = zone16.reset_index()

zone17 = load[load['zone_id']==17]
zone17 = zone17.reset_index()

zone18 = load[load['zone_id']==18]
zone18 = zone18.reset_index()

zone19 = load[load['zone_id']==19]
zone19 = zone19.reset_index()

zone20 = load[load['zone_id']==20]
zone20 = zone20.reset_index()

#Mapping the Stations and the zones
mapping1 = pd.concat((zone1, station4), axis='columns')
mapping1 = mapping1.reset_index()

mapping2 = pd.concat((zone2, station4), axis='columns')
mapping2 = mapping2.reset_index()

mapping3= pd.concat((zone3, station8), axis='columns')
mapping3 = mapping3.reset_index()

mapping4 = pd.concat((zone4, station4), axis='columns')
mapping4 = mapping4.reset_index()

mapping5 = pd.concat((zone5, station8), axis='columns')
mapping5 = mapping5.reset_index()

mapping6 = pd.concat((zone6, station8), axis='columns')
mapping6 = mapping6.reset_index()

mapping7 = pd.concat((zone7, station8), axis='columns')
mapping7 = mapping7.reset_index()

mapping8 = pd.concat((zone8, station8), axis='columns')
mapping8 = mapping8.reset_index()

mapping9 = pd.concat((zone9, station8), axis='columns')
mapping9 = mapping9.reset_index()

mapping10 = pd.concat((zone10, station8), axis='columns')
mapping10 = mapping10.reset_index()

mapping11 = pd.concat((zone11, station8), axis='columns')
mapping11 = mapping11.reset_index()

mapping12 = pd.concat((zone12, station4), axis='columns')
mapping12 = mapping12.reset_index()

mapping13 = pd.concat((zone13, station3), axis='columns')
mapping13 = mapping13.reset_index()

mapping14 = pd.concat((zone14, station8), axis='columns')
mapping14 = mapping14.reset_index()

mapping15 = pd.concat((zone15, station8), axis='columns')
mapping15 = mapping15.reset_index()

mapping16 = pd.concat((zone16, station8), axis='columns')
mapping16 = mapping16.reset_index()

mapping17 = pd.concat((zone17, station8), axis='columns')
mapping17 = mapping17.reset_index()

mapping18 = pd.concat((zone18, station4), axis='columns')
mapping18 = mapping18.reset_index()

mapping19 = pd.concat((zone19, station4), axis='columns')
mapping19 = mapping19.reset_index()

mapping20 = pd.concat((zone20, station8), axis='columns')
mapping20 = mapping20.reset_index()

features = pd.concat((mapping1, mapping2, mapping3, mapping4, mapping5, mapping6, mapping7, mapping8, mapping9, mapping10, mapping11,
                        mapping12,mapping13, mapping14, mapping15, mapping16, mapping17, mapping18, mapping19, mapping20), axis = 0)
features = features.drop(['level_0', 'index', 'daily_load'], axis='columns')

features['date'] = pd.to_datetime(features['date'], errors='coerce')

X = pd.DataFrame(features[['date','zone_id','h1', 'h2', 'h3','h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10',
                           'h11', 'h12','h13', 'h14','h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'h24']])
Y = pd.DataFrame(features[['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10','H11', 'H12', 'H13', 'H14', 'H15', 'H16',
                           'H17', 'H18', 'H19', 'H20','H21', 'H22', 'H23', 'H24','date']])

#Splitting the dataset into training and testing data
x_train = X[(X['date'].dt.year >= 2004) & (X['date'].dt.year <= 2006)]
y_train = Y[(Y['date'].dt.year >= 2004) & (Y['date'].dt.year <= 2006)]
x_test = X[(X['date'].dt.year >= 2007) & (X['date'].dt.year <= 2008)]
y_test = Y[(Y['date'].dt.year >= 2007) & (Y['date'].dt.year <= 2008)]

x_train.drop('date', axis='columns', inplace = True)
x_test.drop('date', axis='columns', inplace = True)
y_train.drop('date', axis='columns', inplace = True)
y_test.drop('date', axis='columns', inplace = True)

print(X.shape)
print(Y.shape)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#Implementing the training models

model1 = DecisionTreeRegressor(random_state = 0)
model1.fit(x_train,y_train)
print("Score Tree: ", model1.score(x_test,y_test))
joblib.dump(model1, 'DecisionTreeRegressor_model.joblib')

model2 = KNeighborsRegressor()
model2.fit(x_train,y_train)
print("Score N_neighbors: ",model2.score(x_test,y_test))


#Extracting new data for prediction
pred = pd.read_csv('Temp_history_final.csv')

x_new_zone1 = pred.iloc[6473:6480,1:]
x_new_zone1.insert(0, 'zone_id','1', True)
x_new_zone2 = pred.iloc[6473:6480,1:]
x_new_zone2.insert(0, 'zone_id','2', True)
x_new_zone3 = pred.iloc[17813:,1:]
x_new_zone3.insert(0, 'zone_id','3', True)
x_new_zone4 = pred.iloc[6473:6480,1:]
x_new_zone4.insert(0, 'zone_id','4', True)
x_new_zone5 = pred.iloc[17813:,1:]
x_new_zone5.insert(0, 'zone_id','5', True)
x_new_zone6 = pred.iloc[17813:,1:]
x_new_zone6.insert(0, 'zone_id','6', True)
x_new_zone7 = pred.iloc[17813:,1:]
x_new_zone7.insert(0, 'zone_id','7', True)
x_new_zone8 = pred.iloc[17813:,1:]
x_new_zone8.insert(0, 'zone_id','8', True)
x_new_zone9 = pred.iloc[17813:,1:]
x_new_zone9.insert(0, 'zone_id','9', True)
x_new_zone10 = pred.iloc[17813:,1:]
x_new_zone10.insert(0, 'zone_id','10', True)
x_new_zone11 = pred.iloc[17813:,1:]
x_new_zone11.insert(0, 'zone_id','11', True)
x_new_zone12 = pred.iloc[6473:6480,1:]
x_new_zone12.insert(0, 'zone_id','12', True)
x_new_zone13 = pred.iloc[4854:4860,1:]
x_new_zone13.insert(0, 'zone_id','13', True)
x_new_zone14 = pred.iloc[17813:,1:]
x_new_zone14.insert(0, 'zone_id','14', True)
x_new_zone15 = pred.iloc[17813:,1:]
x_new_zone15.insert(0, 'zone_id','15', True)
x_new_zone16 = pred.iloc[17813:,1:]
x_new_zone16.insert(0, 'zone_id','16', True)
x_new_zone17 = pred.iloc[17813:,1:]
x_new_zone17.insert(0, 'zone_id','17', True)
x_new_zone18 = pred.iloc[6473:6480,1:]
x_new_zone18.insert(0, 'zone_id','18', True)
x_new_zone19 = pred.iloc[6473:6480,1:]
x_new_zone19.insert(0, 'zone_id','19', True)
x_new_zone20 = pred.iloc[17813:,1:]
x_new_zone20.insert(0, 'zone_id','20', True)

x_new_final = pd.concat((x_new_zone1,x_new_zone2,x_new_zone3,x_new_zone4,x_new_zone5,x_new_zone6,x_new_zone7,x_new_zone8,x_new_zone9,x_new_zone10,
                    x_new_zone11,x_new_zone12,x_new_zone13,x_new_zone14,x_new_zone15,x_new_zone16,x_new_zone17,x_new_zone18,x_new_zone19,x_new_zone20,), axis = 0)

x_new_final2 = pd.DataFrame(x_new_final[['zone_id','h1', 'h2', 'h3','h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10',
                            'h11', 'h12','h13', 'h14','h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'h24']])

print(x_test)
Predicted_test_loads = predictload(x_test)

#Predicting new loads based on temperature data
Predicted_Loads = predictload(x_new_final2)

Predicted_Loads = pd.DataFrame(Predicted_Loads)
x_new_final_extract = x_new_final.iloc[0:,0:4]
x_new_final_extract = x_new_final_extract.reset_index()

x_new_final_extract = pd.DataFrame(x_new_final_extract)
print(Predicted_Loads)
print('++++++++++++++++++++++++++++++')
print(x_new_final_extract)
write_to_file = pd.concat((x_new_final_extract,Predicted_Loads),axis='columns')
write_to_file = write_to_file.drop(['index'], axis='columns')

pd.DataFrame(write_to_file).to_csv('Load_prediction.csv')

