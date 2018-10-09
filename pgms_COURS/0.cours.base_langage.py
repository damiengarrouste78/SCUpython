#!/usr/local/bin/python
# -*- coding: utf-8 -*-

print("généralement, 1 instruction s'écrit sur 1 seule ligne") # commentaire : le reste de la ligne après "#" 

# il est possible de mettre plusieurs instructions sur la même ligne, si elles sont séparées par un point-virgule 
deux_plus_trois = 2 + 3; quatre_plus_cinq = 4 + 5
quatre_plus_cinq # appel de la variable
 
for j in [1, 2, 3, 4, 5]:       # déclaration de bloc par ":"
    print(j)                     # 1ère instruction du bloc j
    for J in [1, 2, 3, 4, 5]:
        print(J)                 # 1ère instruction du bloc J
        print (j + J)             # dernière instruction du bloc J
    print (j)                     # dernière instruction du bloc j
print ('looping done')            # instruction hors bloc

# Rmq 1- Python est sensible à la casse : j et J sont bien 2 variables différentes
# Rmq 2- Les blocs de code sont identifiés par ":" et par l’indentation
#       --> par convention, on fixe l'indentation à 4 espaces
#       --> pratique : paramètrer votre éditeur à 1 tabulation = 4 espaces
#           puis demander à insérer les espaces à la place des tabulations
# Rmq 3- Pour plus de lisibilité l'indentation est ignorée à l'intérieur de (), [] ou {}

tres_long_sur_plusieurs_lignes = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10
                                + 11 + 12 + 13 + 14 + 15 + 16 + 17 + 18 + 19 + 20)

liste_de_liste_facile_a_lire = [[1,2,3],
                                [4,5,6],
                                [7,8,9]]

# une instruction peut aussi s'étendre sur plusieurs lignes si elle finit par un antislash
# dans ce cas, l'indentation est ignorée
deux_plus_trois = 2 + \
                    3


'je suis une chaine' "Moi aussi"
''' chaine
sur plusieurs
lignes'''
r'chaine raw', b'chaine ASCII', u'chaine unicode'

d = (3, False, None, 'chat')
type(d)
print(d[3])

# immuable 'tuple' object does not support item assignment
d[3]='chien'

d = {3: 'Trois', False: 0, None: 'NULL', 'c':'chat'}
type(d)
print(d['c'])

# dict mutable
d['c']='chien'

# nombres
a = float('nan')
a
type(a)

type('inf')

a = float('inf'); a
type(a)
a = float('-inf'); a
type(a)

t = float('NaN')
t

r = r'chaine raw'
type(r)

u = u'chaine unicode'
type(u)
 type(u)

b = b'chaine ASCII'

type(b)

c = 'chaîne'

c

c = u'chaîne'

 c


