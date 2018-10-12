# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 14:49:45 2015
http://blog.yhat.com/posts/predicting-customer-churn-with-sklearn.html
@author: Damien
"""
%reset
#############################################################################################
# Librairies
#############################################################################################

import pandas as pd
import numpy as np
#import pylab as pl
import sys
sys.path.append("C:\\Users\\dgr\\Documents\\Formations\\SCU Python\\NEW\\modules")
sys.path
#import module_lift as mdl
import os
repertoire = "C:\\Users\\dgr\\Documents\\Formations\\SCU Python\\NEW\\data"
os.chdir(repertoire)
#############################################################################################
# Data : score attrition Telco
#############################################################################################

_churn_df = pd.read_csv('churn.csv')
_churn_df.dtypes
# nb de lignes 3333
_churn_df.shape[0]
# nb de vars 21
_churn_df.shape[1]
_churn_df.info()
_churn_df.columns
churn_df=_churn_df
churn_df['Area Code']=churn_df['Area Code'].astype(str)
stats = churn_df.describe(include='all')
# On récupère les colonnes et on les stocke dans une liste
col_names = churn_df.columns.tolist()
print( "Column names:")
print(col_names)

# 6 prem, 6 der colonnes pour visualiser 
#to_show = col_names[:6] + col_names[-6:]
#
#print("Sample data:")
#churn_df[to_show].head(6)

# Création d'un vecteur y contenant la variable cible binarisée
y = np.array((churn_df['Churn?'] == 'True.'))
# ou
#y = np.where(churn_df['Churn?'] == 'True.',1,0)
y.mean()
# 14.5% de churners
pd.crosstab(y, columns = 'count')
#	En %
pd.crosstab(y, columns = 'count', normalize=True)
# col area
print(pd.crosstab(churn_df['Area Code'],y, normalize='index'))
#print(pd.crosstab(churn_df['Area Code'],y).apply(lambda r: r/r.sum(), axis=1))
# flat mais pour la pédagogie on va quand même l'encoder

# par contre les deux var qui indiquent si le client paie pour un service international ou VM sont très discriminantes
 print(pd.crosstab(churn_df["Int'l Plan"],y, normalize='index'))
 print(pd.crosstab(churn_df["VMail Plan"],y, normalize='index'))

 #  les dummies
pd.get_dummies(churn_df['Area Code'],prefix='area_', drop_first=True)
# jointure
churn_df = pd.concat([churn_df,pd.get_dummies(churn_df['Area Code'],prefix='area_', drop_first=True)],axis=1)
#  drop l'original
churn_df.drop(['Area Code'],axis=1, inplace=True)
 
del _churn_df , repertoire
#from sklearn.preprocessing import OneHotEncoder
#onehot_encoder = OneHotEncoder(sparse=False)
#area_2d = np.array(churn_df['Area Code']).reshape(len(churn_df['Area Code']), 1)
#area = onehot_encoder.fit_transform(area_2d)

# étude du state
print(pd.crosstab(churn_df['State'],y))
print(pd.crosstab(churn_df['State'],y,normalize='index'))

#print(pd.crosstab(churn_df['State'],y).apply(lambda r: r/r.sum(), axis=1))
len(churn_df['State'].unique())
# 51 valeurs
# soit on garde et on crée des dummy
# soit on fait une stratégie qui consiste à remplacer la modalité par le taux de 1
# par contre il faut le faire sur l'ech dapp sinon on triche

#############################################################################################
# Partition Apprentissage Test
#############################################################################################
# split Apprentissage Test
from sklearn.model_selection import train_test_split 
state_train, state_test, y_train, y_test = train_test_split(churn_df['State'],y, test_size=0.4)
taux_reponse_state_train=pd.crosstab(state_train,y_train).apply(lambda r: r/r.sum(), axis=1)[1]
taux_reponse_state_test=pd.crosstab(state_test,y_test).apply(lambda r: r/r.sum(), axis=1)[1]
del state_test,y_test,  taux_reponse_state_test
del state_train
# on remplace le state par le taux de reponse associé
list(taux_reponse_state_train)
churn_df['State'].replace(list(taux_reponse_state_train.index), list(taux_reponse_state_train), inplace=True)
churn_df.rename(columns={"State": "churnMoy_state"},inplace=True)

# X
to_drop=['Phone','Churn?']
churn_X = churn_df.drop(to_drop,axis=1)
churn_X.describe()
# 'yes'/'no' has to be converted to boolean values
# NumPy converts these from boolean to 1. and 0. later
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_X[yes_no_cols] = churn_X[yes_no_cols] == 'yes'

# supprime les objets intermédiaires
del to_drop,yes_no_cols

# Stocke les noms des X 
features = list(churn_X.columns)
churn_X.dtypes

# Normalisation 
from sklearn.preprocessing import StandardScaler

# centrer réduire (données doivent être float et pas int)
norm=StandardScaler().fit(churn_X.astype(float))

# on applique le centrage réduction
X = norm.transform(churn_X.astype(float))

# on aurait pu écrire : X=StandardScaler().fit_transform(churn_X.astype(float))

# X en dataframe
X = pd.DataFrame(X)
# on récupère les noms de col
X.columns = churn_X.columns

print ("Feature space holds %d observations and %d features" % X.shape)
print ( "Unique target labels:", np.unique(y))

del taux_reponse_state_train,churn_X,y_train

#############################################################################################
# Partition Apprentissage Test
#############################################################################################
# split Apprentissage Test
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4)

#############################################################################################
# Analyse distribution des variables
#############################################################################################
# 14.4% de churners
y.mean()
X.describe()

#concatenation X et y pour utiliser la fonction boxplot
Xy = pd.concat([X,pd.DataFrame(y)], axis = 1)
# noms des variables 
Xy.columns=features+['Churn?']

#%matplotlib qt
#box plot
#Xy.boxplot(column = ['Account Length'], by = 'Churn?')
Xy.boxplot(column = features[0:6], by = 'Churn?')
Xy.boxplot(column = features[6:12], by = 'Churn?')
Xy.boxplot(column = features[12:], by = 'Churn?')

# tri croisé des var quali
for i in ("Int'l Plan","VMail Plan","area__415","area__510"):
     print(pd.crosstab(X[""+str(i)],y))
     print(pd.crosstab(X[""+str(i)],y).apply(lambda r: r/r.sum(), axis=1))
