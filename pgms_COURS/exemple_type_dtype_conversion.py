# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:39:48 2018

@author: dgr
"""

#lire un fichier csv
import pandas as pd
iris = pd.read_csv("irisHTTP.csv",sep=',',header=None,encoding='latin-1')

# on définit les colonnes
iris.columns = ['petalLength', 'petalWidth', 'sepalLength', 'sepalWidth', 'species']

# structure
iris.info()
iris.dtypes

iris.dtypes
espece = iris.species
espece = iris.species.values.astype(str)
iris.species = iris.species.astype('category')
iris.species.value_counts()
# on voit que le mem usage est plus faible




# arrondir et convertir en integer
petallength=iris.petalLength.round(0).astype(int)
petallength.dtypes
# matrice en format numérique pour que les fonctions marchent
matNum=irismat[:,0:4].astype(float)

# fonction de conversion
x = 1
# 1
x = str(x)
# '1'
x = int(x)
# 1
x = float(x)
# 1.0
x = bool(x)
# True
x = int(x)
# 1
# ne marche pas
y = 'True'
y = int(y)
# marche
y = bool(y)
y = int(y)


# char to date
from datetime import datetime,date
char_date = 'Apr 1 2015 1:20 PM' 
date_obj = datetime.strptime(char_date, '%b %d %Y %I:%M %p') 

# sans espace
char_date = 'Apr1 2015 1:20PM' 
datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b%d %Y %I:%M%p')

# method et attribut d'un objet date
datetime_object.date()
datetime_object.month

# définir une date
date_deb_etude = date(2018,6,1)
date_deb_etude = date(today.year,6,1)