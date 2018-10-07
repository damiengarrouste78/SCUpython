# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 14:18:52 2018

@author: dgr
"""

customers = [{"uid":1,"name":"John"},
    {"uid":2,"name":"Smith"},
           {"uid":3,"name":"Andersson"},
            ]
print(customers)
del customers

for x in customers:
    print (x["uid"], x["name"])

dictionnaire=[{'model':'reg','coef':0.34},{'model':'ridge','coef':0.4}]
dictionnaire+[{'model':'reg','coef':0.34}]