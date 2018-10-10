# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:59:07 2018

@author: dgr
"""

import matplotlib.pyplot as plt

# Première Représentation histogrammme à partir d’un dataframe
iris.petalLength.plot(kind='hist’)

# Second Représentation - graphique construit de A à Z
# un graphique doit être contenu dans un objet figure initialisé
fig=plt.figure() 
#ajoute un sous graphique, ici un seul = 1 ligne, 1 colonne, id 1
ax = fig.add_subplot(1,1,1) 
# trace un histogramme
ax.hist(iris[‘SepalLength'],bins = 10) 
#titre, axes des x y
ax.set(title=‘distribution’,xlabel=‘x’,ylabel=‘effectif’)

plt.show() 

# Troisième Représentation - graphique nuage de points par catégories
import seaborn as sns
sns.pairplot(x_vars='petalLength', y_vars='petalWidth', data=iris, hue="species", size=5)
