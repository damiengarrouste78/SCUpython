# -*- coding: utf-8 -*-
"""
@author: DGR
"""

from sklearn.feature_extraction import FeatureHasher
# on déclare un hasher qui renvoie 10 valeurs
h = FeatureHasher(n_features=10)
D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]
# on applique la methode Transform à la liste D, cela renvoie une  scipy.sparse matrix
f = h.transform(D)
# 2 lignes
f.shape[0]
# 10 vars
f.shape[1]
print(f)
# affichage de la matrice en format compressé
# (num de obs, num de la var) et valeur variable
(0, 2)        -4.0
(0, 3)        -1.0
(0, 9)        2.0
(1, 3)        -2.0
(1, 4)        -5.0

# affichage en matrice (non compressé)
MATRICE_CREUSE=f.toarray()
# Voici une matrice creuse
print(MATRICE_CREUSE)
# représentation en mode compressé (SPARSE)
from scipy import sparse
#[[ 0.,  0., -4., -1.,  0.,  0.,  0.,  0.,  0.,  2.],
# [ 0.,  0.,  0., -2., -5.,  0.,  0.,  0.,  0.,  0.]])
print(sparse.csr_matrix(MATRICE_CREUSE))
#(0, 2)        -4.0
#(0, 3)        -1.0
#(0, 9)        2.0
#(1, 3)        -2.0
#(1, 4)        -5.0
 
#  
# exemple du hashing
from sklearn.feature_extraction import FeatureHasher


#  raw_data = ['This is the first document.',#'This is the second second document.','And the third one.','Is this the first document?' ]
       #- exemple 2
# liste de posts de forum de certaines catégories
raw_data = fetch_20newsgroups(subset='train', categories=categories).data
# Let’s use it to tokenize and count the word occurrences of a minimalistic corpus of text documents:
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(raw_data)
# 3803 lignes
X.shape[0]
# 47885 vars "mots"
X.shape[1]       
# the default configuration tokenizes the string by extracting words of at least 2 letters. The specific function that does this step can be requested explicitly:
# Each term found by the analyzer during the fit is assigned a unique integer index corresponding to a column in the resulting matrix. This interpretation of the columns can be retrieved as follows:
X.toarray()
