# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:18:12 2020

@author: Prince
"""

import pandas as pd
import numpy as np
import seaborn as sns


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


fullhouse = pd.concat([train, test], 
                      keys = ['train', 'test'], axis = 0)

#MISSING VALUES

missing = fullhouse.isna().sum().reset_index()
missing = missing[missing[0] > 1] #for categorical , we can assume most are actually not there,
#for continous, we can calculate with mean, based on relvant correlated fature
#we can also use KNN imputer for missing categorical


fullhouse['MSZoning'] = fullhouse['MSZoning'].fillna('C') #remove 4 missing MSzoning

#LotFrontAge and type of Street. 

fullhouse.groupby('Street')['LotFrontage'].mean()
#hence impute missing by mean of LotFrontage grouped by type of str eet

def lotfront(row):
    if row['Street'] == 'Grvl':
        return 85.8
    elif row['Street'] == 'Pave':
        return 69


fullhouse['LotFrontage'] = fullhouse.apply(lambda row: lotfront(row), axis = 1)
fullhouse[['Alley', 'MasVnrType', 'BsmtQual', 'Fence', 'FireplaceQu', 'MiscFeature', 'PoolQC', 'BsmtCond', 'BsmtExposure','GarageQual', 'GarageCond']] = fullhouse[['Alley', 'MasVnrType', 'BsmtQual', 'Fence', 'FireplaceQu', 'MiscFeature', 'PoolQC', 'BsmtCond', 'BsmtExposure', 'GarageQual', 'GarageCond']].fillna('None') 
corrs = fullhouse.corr()

fullhouse.groupby('MasVnrType')['MasVnrArea'].mean()
def Masnvr(row):
    if row['MasVnrType'] == 'BrkCmn':
        return 195.48
    elif row['MasVnrType'] == 'BrkFace':
        return 261.6
    elif  row['MasVnrType'] == 'None':
        return 0.8
    elif  row['MasVnrType'] == 'Stone':
        return 239.5

fullhouse['MasVnrArea'] = fullhouse.apply(lambda row: Masnvr(row), axis = 1) #replace mising MasVnrArea by mean of respective MAsVnrType

fullhouse['BsmtHalfBath'] = fullhouse['BsmtHalfBath'].fillna(0)
fullhouse['BsmtFullBath'] = fullhouse['BsmtFullBath'].fillna(0)
 #donotremoveNA

fullhouse[['BsmtFinType1', 'BsmtFinType2']] = fullhouse[['BsmtFinType1', 'BsmtFinType2']].fillna('Unf')
fullhouse[['GarageType', 'GarageFinish']] = fullhouse[['GarageType', 'GarageFinish']].fillna('Unf')
fullhouse['GarageYrBlt'] = fullhouse['GarageYrBlt'].fillna('N/A')
fullhouse = fullhouse.drop(['BsmtFinType2'], axis = 1)

#NEW FEATURES
fullhouse['HouseAge'] = fullhouse['YrSold'] - fullhouse['YearBuilt']  #age of house
fullhouse['remodeltime'] = fullhouse['YrSold'] - fullhouse['YearRemodAdd'] #recently remodeled?

def exterior(row):  #check if more than one exterior material covering
    if row['Exterior1st'] == row['Exterior2nd']:
        return 1
    else:
        return 2 
    
fullhouse['number_of_exteriors'] = 0
fullhouse['number_of_exteriors'] = fullhouse.apply(lambda row: exterior(row), axis = 1)


def redundant_exterior(row):  #second second column to none if same as first exterior
    if row['Exterior1st'] == row['Exterior2nd']:
        row['Exterior2nd'] = 'None'
    else:
        row['Exterior2nd'] =  row['Exterior2nd']
    return row['Exterior2nd']
         
fullhouse['Exterior2nd'] = fullhouse.apply(lambda row: redundant_exterior(row), axis = 1)
fullhouse['TotalSqrFeet'] = fullhouse['1stFlrSF'] + fullhouse['2ndFlrSF']  #totalsqrft
fullhouse.drop(['1stFlrSF', '2ndFlrSF'], axis = 1, inplace=True)  


fullhouse['total_porch'] = fullhouse['OpenPorchSF'] + fullhouse['EnclosedPorch'] + fullhouse['3SsnPorch']
fullhouse.drop(['OpenPorchSF','EnclosedPorch', '3SsnPorch' ], axis =1 , inplace=True)


#GrLiv Area and Totalsqrft are highly correlated (0.99)
fullhouse.drop(['GrLivArea'], axis = 1, inplace=True)

#EDA

#yearbuilt vs price

sns.lineplot(x = 'YearBuilt', y = 'SalePrice',
             data = fullhouse)
#price of houses starting 1900 gradually increase by year built

def ancienthouse(row):
    if row['YearBuilt'] < 1900:
        return 'ancient'
    else:
        return 'modern'

fullhouse['anc/mod'] = 0

fullhouse['anc/mod'] = fullhouse.apply(lambda row: ancienthouse(row), axis = 1)

#however, relatively same median price.

#Zone vs price

mszone_ave_price = fullhouse.groupby('MSZoning')['SalePrice'].mean().reset_index()

sns.stripplot(x = 'MSZoning', y = 'SalePrice', data = mszone_ave_price, 
              hue= 'MSZoning')   #Floating Village and Residential low density top average price, commercial lowest

sns.lineplot(x = 'HouseAge', y = 'SalePrice', data = fullhouse)  #unless ancient, price decreases with age
fullhouse.groupby('MoSold').size() #most house sold in June/July

fullhouse.drop(['Id'], axis = 1, inplace=True)

look_up = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May',
            '06': 'Jun', '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

'''ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(sparse = False),
                                        [1,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,26,27,28,29,30,31,32,37,38,39,40,48,50,52,53,55,58,59,60,64,65,66,70,71,77])],
                                        remainder = 'passthrough')'''


fullhouse.groupby('Neighborhood').size()

y = fullhouse['SalePrice']
fullhouse.drop(['SalePrice'], axis =1, inplace=True)



fullhouse['Utilities'] = fullhouse['Utilities'].fillna('AllPub')
fullhouse['Functional'] = fullhouse['Functional'].fillna('Typ')



'''x = ct.fit_transform(fullhouse)'''

#1,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,26,27,28,29,30,31,32,37,38,39,40,48,50,52,53,55,58,59,60,64,65,66,70,71,78]

fullhouse = pd.get_dummies(fullhouse)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

fullhouse = sc.fit_transform(fullhouse)


x_train = np.array(fullhouse[0:1460,])
x_test = np.array(fullhouse[1460:,])
y_train = np.array(y[:1460])


'''from sklearn.linear_model import Lasso #great, moved to 0.2 from 0.4 Stochastic Gradient Decent

regressor = Lasso(alpha = 3)
regressor.fit(x_train, y_train)


lasso_params = [{'alpha': [1,2,3,0.8,0.5,0.25,0.1]}]
from sklearn.model_selection import GridSearchCV
lgs = GridSearchCV(estimator = regressor,
                   param_grid = lasso_params,
                   scoring='neg_root_mean_squared_error',
                  cv = 6,
                  n_jobs=-1)

lgs = lgs.fit(x_train, y_train)
best_accuracy = lgs.best_score_
best_parameters = lgs.best_params_'''


'''from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=250,
                           min_samples_split = 2,
                           min_samples_leaf=1)
rf.fit(x_train, y_train)'''


#GBR
'''from sklearn.model_selection import GridSearchCV
params = [{'n_estimators': [600, 650, 700]}]

gs = GridSearchCV(estimator = gbr,
                  param_grid=params,
                  scoring='neg_root_mean_squared_error',
                  cv = 6,
                  n_jobs=-1)


gs = gs.fit(x_train, y_train)
best_accuracy = gs.best_score_
best_parameters = gs.best_params_'''

x_test = np.nan_to_num(x_test) #replace NAN with 0 in array
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 10)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_


from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=250,
                           min_samples_split = 2,
                           min_samples_leaf=1)
gbr.fit(x_train, y_train)




Y_pred = gbr.predict(x_test)

test = pd.DataFrame(test['Id'])
test['SalePrice'] = Y_pred





test.to_csv('housingfinal.csv', index=False)





    
    