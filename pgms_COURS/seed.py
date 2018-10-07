# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 15:15:07 2018

@author: dgr
"""
#initialiser la graine pour l'aléa
SEED=2018
np.random.seed(SEED)
# on lance une fois
print(np.random.uniform(0,1,3))
# [0.88234931 0.10432774 0.90700933]
# la deuxieme fois, cela change car la seed n'est plus prise en compte
print(np.random.uniform(0,1,3))
#[0.3063989  0.44640887 0.58998539]
# on refixe
np.random.seed(SEED)
print(np.random.uniform(0,1,3))
#[0.88234931 0.10432774 0.90700933]

# Tirage d'individus
s = np.random.normal(-3,3,1000)

# distribution empirique et estimation de densité

import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 15, density=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()

