# -*- coding: utf-8 -*-
"""
Damien
Astuces de programmation python data
"""

%reset #Vider la mémoire de Python

#############################################################################
# créer des données avec numpy
import numpy as np

help(np.array) # demander de l'aide
# la valeur manquante est géré par  np.nan
Matrice = np.array([[1,2,3],[4,5,6],[7,np.nan,9]])
# la valeur manquante
a = float('nan'); b = np.nan ; 
a2 = str('nan') # ce n'est pas un nan

type(a)==type(b)  # comparaison
np.isnan(Matrice) # val manquantes

np.isnan(Matrice).sum(axis=1) # axis=1 :par ligne on boucle sur éléments des colonnes 
np.isnan(Matrice).sum() # au total
np.isnan(Matrice).sum(axis=0) # axis=0 :par col on boucle sur éléments des lignes

del(b) # supprimer l'objet b
type(Matrice) # type de l'objet
dir(Matrice) # dir liste les attr / methodes d'un objet
Matrice.shape # shape est un attribut
Matrice.ravel() # ravel est une méthode

# création d'une matrice, stockage par défaut en float64, ici int32
Matrice = np.array([[1,2,3],[4,5,6],[7,8,9]],dtype=np.int)
# on applique la méthode mean() à l'objet numpy.ndarray moyenne des colonnes
Matrice.mean(axis=0)
# shape et ndim sont des attributs enregistrés
Matrice.shape ; Matrice.ndim
# python est un langage objet : sur la somme des lignes, on applique une somme
    Matrice.sum(axis=0).sum()
# équivalent à la dernière val de l'array envoyée par cumsum
    Matrice.cumsum()[-1]
    
#############################################################################
# DATES
import pandas as pd
date = pd.Timestamp('30/06/2018')
date.year ; date.month
pd.Timestamp.today() ; pd.Timestamp.now()

#############################################################################
# LIRE ET PARCOURIR DES DONNEES


import os # commandes systèmes pour definir le rép de travail
repertoire = "C:\\Users\\dgr\\Documents\\Formations\\SCU Python\\NEW\\data"
os.chdir(repertoire)
os.getcwd()

# pandas permet de lire un json et de créer une table
iris = pd.read_json("iris.json")
iris.columns # index des col


import wget # téléchargement
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'  
wget.download(url,out="irisHTTP.csv") 

#lire un fichier csv
iris = pd.read_csv("irisHTTP.csv",sep=',',header=None,encoding='latin-1')
# on définit les colonnes, columns est un attribut du df qu'on vient écraser
iris.columns = ['petalLength', 'petalWidth', 'sepalLength', 'sepalWidth', 'species']

# structure
iris.info()
iris.dtypes
# résumé incluant les données quali et quanti
iris.describe(include='all').to_csv("summary_iris.csv")
iris.describe(include='all').to_json("summary_iris.json")
stat=iris.describe(include='all') # les chaines sont typés en object

# data management
iris.drop(['petalLength', 'petalWidth'],axis=1) # drop nouvelle variable
iris.rename(columns={'petalLength':'petaleLongueur'}) # nécessite inplace=True 
iris.rename(columns={'petalLength':'petaleLongueur'},inplace=True)
iris=iris.rename(columns={'petalWidth':'petaleLargeur'}) # ou 
liste_especes=list(iris["species"].unique()) # liste des modalités
# arrondir et convertir en integer
iris.petalLength.round(0).astype(int)
# species est une donnée qualitative, on peut la typer en category
iris.species=iris.species.astype('category')

# extraire le deuxieme mot dans la col species
type(iris['species'].str.split('-')) # => Series en sortie
type(iris['species'].str.split('-')[0]) # => list
iris['species2'] = iris['species'].str.split('-')
# species2 c'est un vecteur de liste
iris['species2'] [0][1] # premiere ligne puis seconde col

# donc pour extraire la deuxième partie de la liste
# la fonction apply applique à toutes les lignes la fonction déclaré dans le lambda
iris['species2'] = iris['species'].str.split('-').apply(lambda x: x[1])
iris['flagvirginica'] = (iris.species2 == 'virginica')
# tableau de fréquence
iris.species2.value_counts()
# pct :applique à chaque ligne la fonction lambda
iris.species2.value_counts().apply(lambda x: x/len(iris))
help(pd.value_counts)
iris.species2.value_counts(normalize=True) # il est plus simple de faire



## COMMENT ACCEDER AUX LIGNES ET AUX COL d'un DATAFRAME: slicing
# sur une dataframe toutes les lignes d'une col, renvoie un objet serie
iris["petalLength"] # ou
iris.petalLength
# pour accéder au ligne du coup , comme c'un objet series on met des crochets
# ici accès par l'index en col et lindice en ligne
print(iris["petalLength"][[0,149]])

# la notation mat ne marche pas sur une DF, mais sur une matrice oui
test=iris[:,"petalLength"] ; test=iris[:,0]  # unhashable type: 'slice'
iris.values[:,0:4] # en matrice : ca marche

# loc réfère aux index et iloc aux indices
iris.loc[0] # première ligne
iris.loc[0,"petalLength"] # première ligne et col petalLength
# par contre iris.loc[0,0] ne marche pas car l'index en col est le nom de la var pas un indice
iris.iloc[0,0]  # ligne 1 , col 1
iris.iloc[:,0]  # ligne toutes, col 1
iris.loc[:,"petalLength"] # ligne toutes, col petalLength

# à noter que quand on slice sur les index le terme de droite est inclus!
iris.loc[0:1,"petalLength"]
# à noter que quand on slice sur les indices le terme de droite est exclus!
iris.iloc[0:1,0]

# pour indiquer des listes, il faut créer une liste entre []
iris.loc[:,['petalLength', 'petalWidth', 'sepalLength', 'sepalWidth']]

# accès en indices
iris.iloc[:,[0,1,2,3]]
iris.iloc[:,range(4)]
# attention, accès en index: avec 2 bornes, la seconde est incluse contrairement aux accès par indices
iris.iloc[0:149,0] # 149 lignes
iris.loc[0:149:,"petalLength"] # 150 lignes
(iris.iloc[0:149,0].shape == iris.loc[0:149:,"petalLength"].shape) # => False