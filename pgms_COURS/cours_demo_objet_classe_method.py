# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 07:05:13 2019

@author: DGA
"""


# génère 6 individus sur 2 variables avec  numpy
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

type(X)
# X est un objet de type array 
# ndarray c'est une classe
# https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray

# cette classe a la methode mean
# on peut mettre la donnee en argument
np.mean(X[0])
# soit mettre objet.methode()
X[0].mean()

# shape est un attribut de la classe ndarray
X.shape


# PCA sur les 6 individus
from sklearn.decomposition import PCA
?PCA
# https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/decomposition/pca.py#L104
# on peut voir la définition de la classe, les attr, les methodes
# la methode interne _fit contient le coeur de l'algorithme


# on déclare la recette : instancier la PCA avec 2 composantes
maPCA=PCA(n_components=2)

# si on demande un attribut à ce niveau, seul le constructeur de la classe a été définie
print(maPCA.n_components)
# les attributs résultats sont calculés à l'execution des méthodes, ici ils n'existent pas
print(maPCA.explained_variance_)


# on cuisine : on apprend la PCA avec la méthode fit
maPCA.fit(X)
# maPCA devient un objet disponible pour être appellé
# on affiche les résultats : attributs
print(maPCA.explained_variance_)
print(maPCA.explained_variance_ratio_) 

# on applique la recette cuisinée sur une nouvelle matière : 1 data point, quelle est sa projection dans le sous espace
new_x = np.array([-1,-0,5])
pred_x=maPCA.transform(new_x)



