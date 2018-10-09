# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 14:18:52 2018

@author: dgr
"""
# pour mettre plusieurs entrées, il faut créer une liste de dictionnaires
customers = [{"uid":1,"name":"John"},
    {"uid":2,"name":"Smith"},
           {"uid":3,"name":"Andersson"},
            ]
print(customers)
type(customers)
del customers

# customer est une liste , c'est un iterable sur lequel on peut boucler
for x in customers:
    print (x["uid"], x["name"])

# exemple utilisation dictionnaire
dictionnaire=[{'model':'reg','coef':0.34},{'model':'ridge','coef':0.4}]
# ajout d'eune entree
dictionnaire+[{'model':'reg2','coef':0.34}]

# afficher une valeur : la deuxieme entree et la valeur du coef
# dictionnaire est une liste donc on accède à la premiere entrée qui est un dict avec l'objet coef
dictionnaire[1]['coef']