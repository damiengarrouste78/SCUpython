# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:46:25 2018

@author: dgr
"""
import numpy as np
def standardisation(x):     
    x-=np.mean(x) # the -= means can be read as x = x- np.mean(x)
    x/=np.std(x)
    return x

# test
# La variable __name__ varie selon le module dans lequel on se trouve durant l'exécution du programme. 
# Dans le module principal, sa valeur sera égale à __main__.
# alors que si le pgm est appellé par import  alors elle ne vaut pas main et ainsi l'instruction suivante
# ne sera pas effectué 
    
if __name__=="__main__":
    # variable aléa de loi uniforme entre 0 et 1
    tab=np.random.uniform(1,2,100)
    standardisation(tab)
    print("la moyenne est elle bien nulle ?{}".format(tab.mean()))
    
    
