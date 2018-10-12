# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:31:50 2018

@author: dgr
"""

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
os.chdir("C:\\Users\\dgr\\Documents\\Formations\\SCU Python\\NEW\\data")
#https://www.kaggle.com/apratim87/housingdata/version/1#housingdata.csv
#Housing data

#Variables in order: CRIM per capita crime rate by town 
#ZN proportion of residential land zoned for lots over 25,000 sq.ft. 
#INDUS proportion of non-retail business acres per town 
#CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
#NOX nitric oxides concentration (parts per 10 million) 
#RM average number of rooms per dwelling 
#AGE proportion of owner-occupied units built prior to 1940 
#DIS weighted distances to five Boston employment centres 
#RAD index of accessibility to radial highways 
#TAX full-value property-tax rate per $10,000 
#PTRATIO pupil-teacher ratio by town 
#B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
#LSTAT % lower status of the population 
#MEDV Median value of owner-occupied homes in $1000's
housing = pd.read_csv('housingdata.csv',header=0,index_col=False,sep=',',
names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX", "PTRATIO","BlackPop","LSTAT","MEDV"])

housing.CHAS=housing.CHAS.astype(bool)
#Housing data
housing.dtypes
# nb de lignes 3333
housing.shape[0]
# nb de vars 21
housing.shape[1]
housing.info()
housing.columns

stats = housing.describe(include='all')

# créer 3 groupes de valeur des logements <15 , 15-25 et 25+
housing["Valeur_Immo"] = '2.15k-25k$'
housing.loc[housing['MEDV'] >= 25, 'Valeur_Immo'] = '3.>25k$'
housing.loc[housing['MEDV'] <15, 'Valeur_Immo'] = '1.<15k$'
housing.Valeur_Immo.describe()
housing.info()
housing.Valeur_Immo=housing.Valeur_Immo.astype('category')
housing.info() # réduction de la mém