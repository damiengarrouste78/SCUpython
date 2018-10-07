# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 12:24:01 2018

@author: dgr
"""

#Vider la mémoire de Python
%reset

# créer des données avec numpy
import numpy as np

# demander de l'aide
help(np.array)
# la valeur manquante est géré par  np.nan
Matrice = np.array([[1,2,3],[4,5,6],[7,np.nan,9]])

a = float('nan')
b = np.nan
# comparaison
type(a)==type(b)

np.isnan(Matrice)
# combien 
np.isnan(Matrice).sum()


# supprimer l'objet b
del(b)
# type de l'objet
type(Matrice)
# dir liste les attr / methodes d'un objet
dir(Matrice)

import pandas as pd
date = pd.Timestamp('30/06/2018')
date.year ; date.month
pd.Timestamp.today() ; pd.Timestamp.now()

# commandes systèmes pour definir le rép de travail
import os
repertoire = "C:\\SCU Python"
os.chdir(repertoire)

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

# LIRE UN FICHIER CSV PROVENANT DU WEB
repertoire = "C:\\Users\\dgr\\Documents\\Formations\\SCU Python\\NEW\\data"
os.chdir(repertoire)
# téléchargement
import wget
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'  
wget.download(url,out="irisHTTP.csv") 

#lire un fichier csv
import pandas as pd
iris = pd.read_csv("irisHTTP.csv",sep=',',header=None,encoding='latin-1')

# on définit les colonnes
iris.columns = ['petalLength', 'petalWidth', 'sepalLength', 'sepalWidth', 'species']

# structure
iris.info()
iris.dtypes
# résumé incluant les données quali et quanti
iris.describe(include='all').to_csv("summary_iris.csv")
iris.describe(include='all').to_json("summary_iris.json")
stat=iris.describe(include='all')
# les chaines sont typés en object

# drop, rename, nouvelle variable
iris.drop(['petalLength', 'petalWidth'],axis=1)
iris.rename(columns={'petalLength':'petaleLongueur'})

#conversion de la liste des valeurs distinctes de species dans une liste
liste_especes=list(iris["species"].unique())

# arrondir et convertir en integer
iris.petalLength.round(0).astype(int)

# species est une donnée qualitative, on peut la typer en category
iris.species.astype('category')
# split de la colonne species ,
type(iris['species'].str.split('-'))
# => Series en sortie
# si on prend une ligne, on voit que le type est list
type(iris['species'].str.split('-')[0])
# => list

# donc pour extraire la deuxième partie de la liste , 
#il faut appliquer une fonction par lignes, 1 réfère au 2e élément

iris['species2'] = iris['species'].str.split('-')
# la fonction apply applique à toutes les lignes la fonction déclaré dans le lambda
iris['species2'] = iris['species'].str.split('-').apply(lambda x: x[1])
iris.dtypes
iris['flagvirginica'] = (iris.species2 == 'virginica')
# tableau de fréquence
iris.species2.value_counts()
# pct :applique à chaque ligne la fonction lambda
iris.species2.value_counts().apply(lambda x: x/len(iris))