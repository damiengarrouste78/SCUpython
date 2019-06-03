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
# on voit que ce qui est dans le main ne s'exécute pas

import os
os.getcwd()
repertoire = "C:\\Users\\dgr\\Documents\\Formations\\SCU Python\\NEW\\data"
os.chdir(repertoire)

# test de la fonction sur une dataframe
import pandas as pd
iris = pd.read_json("iris.json")

# axis=0 on raisonne par col, on prend toutes les lignes de chaq col

# utilisation de la fonction , en entrée on attend 1 vecteur
iris_norm_petal_length= standardisation(iris['petalLength'])
# ma fonction ne marche sur plusieurs vecteurs, donc il faut que j'applique col apres col la fonction
iris_norm = iris[['petalLength', 'petalWidth', 'sepalLength', 'sepalWidth']].apply(standardisation,axis=0)

# par défaut axis=0, on voit que la moy =0
resultat=dict(iris_norm.mean(axis=0))
#iris_norm.mean()



#  en json
import json
# on sauvegarde le dictionnaire avec la librairie json
with open("exemple.json", "w") as f:
    json.dump(resultat, f)
#cette ligne s'affiche dans le sdtout par défaut donc la console en exec batch
print("exécution terminée")