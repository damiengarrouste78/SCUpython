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

# test de la fonction sur une dataframe
import pandas as pnd
iris = pd.read_json("iris.json")

iris_norm = iris[['petalLength', 'petalWidth', 'sepalLength', 'sepalWidth']].apply(standardisation,axis=0)