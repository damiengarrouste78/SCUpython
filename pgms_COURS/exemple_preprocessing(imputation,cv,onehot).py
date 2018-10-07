# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:22:42 2018

@author: dgr
"""



# one hot exemple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
data = np.array(['ZERO', 'ZERO', 'DEUX', 'UN', 'ZERO', 'ZERO', 'DEUX', 'UN', 'DEUX', 'ZERO'])
# integer encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data)
# binary encoding : le one hot attend un  Expected 2D array
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

onehot_encoder = OneHotEncoder(sparse=True)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

# invert first example
from numpy import argmax
#inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X =np.array([[1, 2], [np.nan, 3], [7, np.nan]])
print(imp.fit_transform(X))                           

np.isnan(X)
# combien 
np.isnan(X).sum()
pd.isnull(pd.DataFrame(X,columns=['x1','x2'])).sum() 

# par une valeur
don = pd.DataFrame(X,columns=['x1','x2'])
don['x1']=don.fillna(0)
# par un calcul
don['x2']=don.fillna(don['x2'].mean())

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_encoder = LabelEncoder()
# on numérise (integer encoding)
integer_encoded = label_encoder.fit_transform(iris.target)
# on ajoute une dimension car le one hot attend un  Expected 2D array
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# on applique le one hot encoding
onehot_encoder = OneHotEncoder(sparse=False)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)


