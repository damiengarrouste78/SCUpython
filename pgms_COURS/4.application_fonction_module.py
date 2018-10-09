# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:44:23 2018

@author: dgr
"""

# on déclare un repertoire supplémentaire 
# où python peut aller chercher des définitions de modules
import sys
sys.path.append("C:\\Users\\dgr\\Documents\\Formations\SCU Python\\NEW\modules")
sys.path



# importation d'une fonction stockée en module
from standardisation import standardisation
#import standardisation as std


import os
os.getcwd()
repertoire = "C:\\Users\\dgr\\Documents\\Formations\\SCU Python\\NEW\\data"
os.chdir(repertoire)

# test de la fonction sur une dataframe
import pandas as pd
iris = pd.read_json("iris.json")

# axis=0 on raisonne par col, on prend toutes les lignes de chaq col
iris_norm = iris[['petalLength', 'petalWidth', 'sepalLength', 'sepalWidth']].apply(standardisation,axis=0)

# par défaut axis=0, on voit que la moy =0
iris_norm.mean(axis=0)
iris_norm.mean()