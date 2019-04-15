# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:03:05 2018

@author: dgr
"""

# on travaille sur le célèbre jeu IRIS
import pandas as pd
import numpy as np
import os

#os.chdir("D:\\SCU Python\\NEW\\data")
os.chdir("C:\\Users\\dgr\\Documents\\Formations\SCU Python\\NEW\data")
os.getcwd()

iris = pd.read_json("iris.json")
iris.columns

type(iris)
# pandas.core.frame.DataFrame
# type pandas des col
iris.dtypes

type(iris.petalLength)
# pandas.core.series.Series

iris = iris[['sepalLength', 'sepalWidth','petalLength', 'petalWidth','species']]
#Analyser le temps de traitement d’une cellule 

# centrer réduire les données
from sklearn.preprocessing import StandardScaler
test = StandardScaler().fit_transform(iris[['sepalLength', 'sepalWidth','petalLength', 'petalWidth']])
#test.mean(axis=0)
# Transforme les donnéesen ndarray
irismat=iris.as_matrix()
# ou Use .values instead.
irismat=iris.values

# matrice en format numérique pour que les fonctions marchent
matNum=irismat[:,0:4].astype(float)

# centrer réduire les données
from sklearn.preprocessing import StandardScaler
matNum = StandardScaler().fit_transform(matNum)

# Un peu de Mathématique  ACP : diagonalisation de la matrice des corrélations

matCorr = np.corrcoef(matNum,rowvar = False) # matrice des corrélations, rowvar =F signifie que les col sont en col
# diagonalisation fonction eig donne  vecteur de valeurs propres et un array pour  les vecteurs propres, 1 par colonne
eigenVal, eigenVec = np.linalg.eig(matCorr)

#eigenValVec = np.linalg.eig(matCorr)

# aide : that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
eigenVec1 = eigenVec[1,] # vecteur propre Axe 1

# projection dans l'espace de l'ACP
# formule pour un individu sur un axe u : combinaison linéaire entre les valeurs centrées réduites et le vecteur propre u de l'axe
 
# créons une fonction qui standardise une colonne (1darray)
def standardisation(x):     
    x-=np.mean(x) # the -= means can be read as x = x- np.mean(x)
    x/=np.std(x)
    return x

# test de la fonction sur une data frame
iris_norm = iris[[ 'sepalLength', 'sepalWidth','petalLength', 'petalWidth']].apply(standardisation,axis=0)
iris_norm.mean() # moy = 0
# test de la fonction sur une colonne d'un array
standardisation(matNum[:,0]).shape # 150 lignes
standardisation(matNum[:,0]).mean() # moy =0 
standardisation(matNum[:,0]).std() # std =1
  
  # appliquer une fonction le long des colonnes d'un 2darray
matCR =  np.apply_along_axis(standardisation,0,matNum)
  
matCR.mean(axis=0) # verif 0
matCR.std(axis=0)  # verif 1
  
# la projection dans l'axe i est la combinaison linéaire entre le vecteur centré réduit de l'individu et le vecteur propre
prin1_individu1 =  np.dot( matCR[1,:] , eigenVec[,1].transpose() )
prin_individu1 =   np.dot( matCR[1,:] , eigenVec[:,:].transpose() )
# les deux premiers vecteurs propres
eigenVec[:,0:2]
# la projection des 150 individus sur les 2 premiers axes
prin14_individu150 =   np.dot( matCR , eigenVec[:,0:4] )

  
 # verif avec la fonction de SCIKIT LEARN - projection des 150 individus sur l'axe 1
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Standardizing the features
pca = PCA(n_components=4)

# projection des individus dans 4 axes en une ligne de code
principalComponents = pca.fit_transform(StandardScaler().fit_transform(matNum))

# en plusieurs étapes
centreereduite=StandardScaler().fit_transform(matNum)
pca = pca.fit(centreereduite)
cp = pca.transform(centreereduite)
