# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# PROJET : Formation interne - Python pour la Data Science
# OBJET : Exercices fondamentaux
# DATE : 2016-06
# AUTEUR : Soft Computing - GLC
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# 1. MANIPULATION DE CHAINES
#------------------------------------------------------------------------------
#
# 1.1. déclarer les 3 chaines suivantes :
#       - Hello World
#       - I'm here !
#       - Python is a good programming language !
# les 2 premières chaines seront déclarées normalement, sur une seule ligne
# la dernière chaine sera déclarée sur 3 lignes

s1 = 'Hello World'
s2 = "I'm here !"
s3 = """Python is a good
programming
language !"""


# 1.2. afficher la concaténation des 2 premières chaines,
#   en insérant la ponctuation necessaire pour la render lisible
print s1 + ", " + s2

# 1.3. afficher 4 répétitions de la 1ère chaine, en insérant un séparateur
print (s1+", ")*4

# 1.4. afficher la 3ème chaine, puis sa longueur, et ce en 1 seule instruction
print s3, len(s3)

# 1.5. en utilisant le formatage, afficher l'insertion de la première chaine dans
# la chaine suivante : 'I told you : '   
print "I told you : %s" % s1

# 6. par indiçage, extraire la sous-chaine 'a good programming' de la 3ème chaine,
#   en partant du début, puis en partant de la fin
s3[10:28]
s3[-29:-11]

# 1.7. déclarer 3 variables, contenant respectivement votre nom, prénom et profession
#   en utilisant le formatage, afficher une chaine de la forme suivante :
#   print "nom : {}, prénom: {}, profession : {}".format(profession, nom, prenom)

nom = "GARROUSTE"
prenom = "Damien"
profession = "Data Scientist"

# completer l'instruction en accédant aux arguments par la position
print "nom : {}, prénom: {}, profession : {}".format(profession, nom, prenom)

# completer l'instruction en accédant aux arguments par le nom
print "nom : {n}, prénom: {p}, profession : {w}".format(w=profession, n=nom, p=prenom)


#------------------------------------------------------------------------------
# 2. MANIPULATION DE LISTES ET TUPLES
#------------------------------------------------------------------------------

# 2.1. créer une 1ère liste de valeurs de 2 à 18 avec un pas de 2 (sans tout lister manuellement)
l1 = range(2,20,2)

# 2.2. créer une 2ème liste dont les valeurs sont celles de la 1ère divisée par 2
#       attention à conserver le format des valeurs de la 1ère liste 
l2 = [int(x/2) for x in l1]

# 2.3. sommer les valeurs des 2 listes créées dans une 3ème (utiliser la fonction zip)
l3 = [x + y for x, y in zip(l1, l2)]

# 2.4. retourner la longueur de la 3ème liste, ainsi que l'indice de la valeur 15
len(l3)
l3.index(15)

# 2.5 inverser puis retrier la 3ème liste
l3.reverse()
l3.sort()

# 2.6 supprimer le dernier élément de la 2ème liste puis le réinsérer en 3ème position
#   transformer cette liste en tuple et renouveller l'opération avec la fonction len() et l'indiçage
supp = l2.pop()
l2.insert(2,supp)
t2 = tuple(l2)
t3 = t2[:2] + (t2[len(t2)-1],) + t2[2:(len(t2)-1)]

# 2.7. créer un tableau de 10 lignes et 5 colonnes, dont la valeur d'une cellule correspond
#       à la position dans le tableau, ie position en ligne * position en colonne tel que ci-après
#       ensuite, remplacer la valeur 18 par vide et la valeur 32 par "Hello"
"""
[[1, 2, 3, 4, 5],
 [2, 4, 6, 8, 10],
 [3, 6, 9, 12, 15],
 [4, 8, 12, 16, 20],
 [5, 10, 15, 20, 25],
 [6, 12, 18, 24, 30],
 [7, 14, 21, 28, 35],
 [8, 16, 24, 32, 40],
 [9, 18, 27, 36, 45],
 [10, 20, 30, 40, 50]]
"""
l_lignes = range(0,10)
l_colonnes = range(0,5)
tableau = []
for i in l_lignes:
    tableau.append([])
    for j in l_colonnes:
        tableau[i].append((i+1)*(j+1))
tableau
tableau[5][2] = None
tableau[7][3] = 'Hello'
tableau


#------------------------------------------------------------------------------
# 3. MANIPULATION DE DICTIONNAIRE
#------------------------------------------------------------------------------

# 3.1. créer un dictionnaire à partir des 2 listes suivantes
#   utiliser la 1ère liste comme clés, la 2nde comme valeurs
l1 = ['zéro','un','deux','trois','quatre','cinq']
l2 = [0,1,2,3,4,5]
d= dict(zip(l1,l2))

# 3.2. afficher le dictionnaire, puis la liste des clés, et la liste des valeur
#   retourner la valeur pour la clée 'trois'
#   retourner la clée pour la valeur 3
print d, d.keys(), d.values()
d['trois']
for k, v in d.items():
    if v == 3:
        print "la clé de la valeur 3 est", k

# 3.3. switcher les clés et les valeurs
dinv = {v: k for k, v in d.items()}

# 3.4. ajouter une entrée dans le dictionnaire inversé en continuant la suite
#   modifier la valeur de la clé 2 pour lui affecter sa traduction anglaise,
#       après avoir vérifié que cette clé existe
#   supprimer l'entrée pour la valeur 'quatre'
dinv[6] = 'six'
if dinv.has_key(2):
    dinv[2] = 'two'
for k, v in dinv.items():
    if v == 'quatre':
        del dinv[k]


#------------------------------------------------------------------------------
# 4. MANIPULATION DE FICHIER ET FONCTIONS
#------------------------------------------------------------------------------

#from __future__ import unicode_literals

# 4.1. Lire le fichier Formation_Python_Fichier_Test.csv dans le répertoire Programmes
#       créer une liste pour l'entête, une autre pour les lignes
f = open(u"C:/Users/glc/Documents/1_PROJETS/FORMATION_PYTHON/Programmes/Formation_Python_Fichier_Test.csv", "r")
r = f.readlines()
header = r[0]
lines = r[1:]

# 4.2. créer une fonction qui pour chaque ligne :
#   - supprime les doubles quotes
#   - ne conserve que le prénom, l'ANNEE de naissance et l'âge
#   - avec un séparateur '|'
#   Ecrire le résultat dans un nouveau fichier
#   Lire ce nouveau fichier et vérifier le contenu
import re
def reformat_lines(y):
    y_list = [re.sub('"', '', re.sub('-', ';', y)).split(";")[0],
              re.sub('"', '', re.sub('-', ';', y)).split(";")[3],
              re.sub('"', '', re.sub('-', ';', y)).split(";")[4]]
    return '|'.join(map(str, y_list))

fout = open(u"C:/Users/glc/Documents/1_PROJETS/FORMATION_PYTHON/Programmes/Formation_Python_Fichier_TestOut.csv", "w")
for l in lines:
    l_ok = reformat_lines(l)
    print l_ok
    fout.write(l_ok)
fout.close()

f2 = open(u"C:/Users/glc/Documents/1_PROJETS/FORMATION_PYTHON/Programmes/Formation_Python_Fichier_TestOut.csv", "r")
l2 = f2.readlines()
l2
