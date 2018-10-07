# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 10:41:43 2018

@author: dgr
"""
#%reset
import pandas as pd
import os
repertoire = "C:\\Users\\dgr\\Documents\\Formations\\SCU Python\\NEW\\data"
os.chdir()
iris = pd.read_csv("irisHTTP.csv",sep=',',header=None,encoding='latin-1')
# on définit les colonnes
iris.columns = ['petalLength', 'petalWidth', 'sepalLength', 'sepalWidth', 'species']

import seaborn as sns
corr = iris.corr()
sns.heatmap(corr,xticklabels=corr.columns.values,
yticklabels=corr.columns.values)


# modèle de régression linéaire multiple
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(iris.iloc[:,1:4],iris.petalLength)
# regarder le contenu du modèle
reg.__dict__
# on peut visualiser les paramètres estimés du modèle
# 'coef_': array([ 0.65486424,  0.71106291, -0.56256786])
petalLength_pred = reg.predict(iris.iloc[:,1:4])
# regression métrics
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from math import sqrt 
r2_score(iris.petalLength, petalLength_pred)  
# 0.86
mean_absolute_error(iris.petalLength, petalLength_pred)
# 0.24
sqrt(mean_squared_error(iris.petalLength, petalLength_pred))
# 0.31
sqrt(mean_squared_log_error(iris.petalLength, petalLength_pred))
# 0.045

import numpy as np
# corrélations
#plt.matshow(iris.corr())
import seaborn as sns
corr = iris.corr()
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)

# étude de sepalLength
sns.pairplot(x_vars='petalWidth', y_vars='sepalLength', data=iris, hue="species", size=5)

# on va séparer en deux groupes et étudier que les non setosa

X = iris.loc[iris.species != 'Iris-setosa',['petalLength', 'petalWidth', 'sepalWidth']]
y = iris[iris.species != 'Iris-setosa'].sepalLength

# Régression linéaire simple
# quel est le signe de petal.width pour expliquer sepalLength
reg = linear_model.LinearRegression()

# 2D array nécessaire pour les X même si une seule feature
X_petalWidth=np.array(X['petalWidth']).reshape(len(X.petalWidth), 1)
reg.fit(X_petalWidth,y)
sepalLength_pred =reg.predict(X_petalWidth)

sns.pairplot(x_vars='petalWidth', y_vars='sepalLength', data=iris[iris.species != 'Iris-setosa'], hue="species", size=5)

print('Coefficients: \n', reg.coef_)
# The coefficient : 1.29 * petalWidth (cotnre -1.71 au global)
print("root Mean squared error: %.2f" %sqrt(mean_squared_error(y, sepalLength_pred)))
print('Variance score: %.2f' % r2_score(y, sepalLength_pred))

import matplotlib.pyplot as plt
# nuage de points X, y
plt.figure()
plt.scatter(X_petalWidth, y,  color='black')
plt.plot(X_petalWidth, sepalLength_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())

plt.show()

# rég linéaire multiple sur données corrélées
reg = linear_model.LinearRegression()
reg.fit(X,y)
print(pd.DataFrame({'Coefficients': list(reg.coef_)}, list(X.columns.values)))
#             Coefficients
#petalLength      0.695178
#petalWidth      -0.247895
#sepalWidth       1.066154
#  quand on explique la variation de Y par les 3 var, le signe de Petal.Width devient neg ! -0.25
# CONCLUSION : la rég linéaire donne des résultats biaisés en présence de COLLINEARITE (corrélations multivariées)
print('Variance score: %.2f' % r2_score(y, reg.predict(X)))
# 0.86

# Si on enleve une variable
reg = linear_model.LinearRegression()
reg.fit(X[["sepalWidth","petalLength"]],y)
print(pd.DataFrame({'Coefficients': list(reg.coef_)}, list(X[["sepalWidth","petalLength"]].columns.values)))
#             Coefficients
#sepalWidth       0.994963
#petalLength      0.653339
print('Variance score: %.2f' % r2_score(y, reg.predict(X[["sepalWidth","petalLength"]])))


# la régression ridge permet de garder tous les prédicteurs en evitant que les coeff
# prennent des valeurs contradictoires pour compenser les corrélations
from sklearn import linear_model
ridge = linear_model.Ridge (alpha = 100)
ridge.fit(X,y)
print(pd.DataFrame({'Coefficients': list(ridge.coef_)}, list(X.columns.values)))


from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])       
RidgeCV(alphas=[0.1, 1.0, 10.0], cv=None, fit_intercept=True, scoring=None,
    normalize=False)
reg.__dict__
reg.alpha_                                      
0.1