# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:04:04 2018

@author: dgr
"""

# on crée un tuple
numbers = (1, 2, 3, 4)

# utilise la lambda func pour à tout x renvoyer f(x)=x²
# map applique une fonction à un iterable, ici le tuple
result = list(map(lambda x: x*x, numbers))

# autre fonction
result= list(map(lambda x: x==1.00, numbers))
print(result)
result.count(False)


# fonction en dur avec gestion des exceptions
from math import sqrt
def _sqrt(x):
    try:
        x>=0
        return sqrt(x)
    except:
        x<0
        print("calcul impossible")
    
    # test
    _sqrt(-5)
    _sqrt(+5)
    
    
result=list(map(_sqrt, numbers))  # Output [2, 4, 6, 8]
print(result)