# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 08:31:58 2018

@author: dgr
"""
# charge la librairie numpy : librairie de manip matrice
import numpy as np

# créer un objet array qui s'appellera matrice, 4 lignes et 3 col,
# données stockées en integer
matrice = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],dtype=int)
#matrice = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],dtype=float)
type(matrice)
# matrice est un objet de données 
# on peut lui appliquer des fonctions/méthodes 
matrice.cumsum(axis=1)

# j'affiche l'attribut shape de l'objet matrice
dimensions=matrice.shape
# ca renvoie un résultat qui lui meme va 
type(dimensions)
#shape n'est pas une fonction, mais un attribut qui renvoie une liste
# donc pour y accéder c l'opérateur []
nbcol=matrice.shape[1]


matrice.cumsum(axis=1)
# somme des lignes 
matrice.cumsum(axis=1)[:,nbcol-1]

matrice.cumsum(axis=1)[:,nbcol-1].sum()

%reset

del matrice,nbcol