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
prin14_individu150 =   np.dot( matCR , eigenVec[:,0:2] )

  
 # verif avec la fonction de SCIKIT LEARN - projection des 150 individus sur l'axe 1
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Standardizing the features
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(StandardScaler().fit_transform(matNum))

centreereduite=StandardScaler().fit_transform(matNum)
pca = pca.fit(centreereduite)
cp = pca.transform(centreereduite)

## COMMENT ACCEDER AUX LIGNES ET AUX COL d'un DATAFRAME: slicing

# sur une dataframe si on veut toutes les lignes d'une col, renvoie un objet serie
iris["petalLength"]
# ou
iris.petalLength
# pour accéder au ligne du coup , comme c'un objet series on met des crochets
# ici accès par l'index en col et lindice en ligne
print(iris["petalLength"][[0,149]])

# la notation mat ne marche pas sur une DF, mais sur une matrice oui
test=iris[:,"petalLength"] ; test=iris[:,0] 
# unhashable type: 'slice'

# autre méthode loc et iloc
# loc réfère aux index et iloc aux indices

# première ligne
iris.loc[0]
# première ligne et col petalLength
iris.loc[0,"petalLength"]
# par contre iris.loc[0,0] ne marche pas car l'index en col est le nom de la var pas un indice
# ligne 1 , col 1
iris.iloc[0,0] 
# ligne toutes, col 1
iris.iloc[:,0] 
# ligne toutes, col petalLength
iris.loc[:,"petalLength"]

# à noter que quand on slice sur les index le terme de droite esst inclus!!
iris.loc[0:1,"petalLength"]
# à noter que quand on slice sur les indices le terme de droite esst exclus!!!!
iris.iloc[0:1,0]

# pour indiquer des listes, il faut créer une liste entre []

# iris.columns renvoie une liste de toutes les var
iris.loc[:,['petalLength', 'petalWidth', 'sepalLength', 'sepalWidth']]

# accès en indices
iris.iloc[:,[0,1,2,3]]
iris.iloc[:,range(4)]
# attention, pour les accès en index :  avec 2 bornes, la seconde est incluse contrairement aux accès par indices
iris.iloc[0:149,0] # 149 lignes
iris.loc[0:149:,"petalLength"] # 150 lignes
(iris.iloc[0:149,0].shape == iris.loc[0:149:,"petalLength"].shape)
# => False
