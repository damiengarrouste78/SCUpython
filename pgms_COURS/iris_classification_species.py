# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 08:48:53 2018

@author: dgr
"""



# Apprentissage de modèle SVM avec évaluation des métriques sur CV 4 fold)
from sklearn import svm, model_selection , datasets
iris = datasets.load_iris()
model = svm.SVC(kernel='linear', C=0.1)
# le modele SVM est un classifier pour séparer les catégories de species
scores = model_selection.cross_validate(model,iris.data,iris.target,scoring='accuracy',cv=4,return_train_score=True)
# le resultat contient pour chacun des 4 fold La métrique sur le fold laissé de coté : métrique cross validée
#scores.keys()
scores['test_score']
# array([1.        , 0.97435897, 0.97222222, 0.97222222])
scores['train_score']
# array([0.97297297, 0.99099099, 0.99122807, 0.98245614])
scores['train_score'].mean() - scores['test_score'].mean()
# 0.0047 : les erreurs appr vs CV sont proches ce qui milite pour garder ce classifier


from sklearn import datasets,grid_search,svm
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score,roc_auc_score
from sklearn.model_selection import train_test_split ,GridSearchCV
iris = datasets.load_iris()
# GRILLE : on fait varier le noyau et la marge
parameters = {'kernel':('linear', 'rbf'), 'C':[0.01,0.05,0.1,0.25,0.5,1]}
cv_model = GridSearchCV(svm.SVC(), parameters,scoring='accuracy')
cv_model.fit(iris.data, iris.target)
cv_model.best_params_
#  {'C': 0.5, 'kernel': 'linear'}
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
model=svm.SVC(C=0.5,kernel='linear',probability=True)
model.fit(X_train, y_train)
proba=model.predict_proba(X_train)
proba.shape
# 3 colonnes car 3 catégories à prédire
species_pred=model.predict(X_train)
confusion_matrix(y_train, species_pred)
accuracy_score(y_train, species_pred)
accuracy_score(y_test, model.predict(X_test))
f1_score(y_train, species_pred,average='weighted')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k')
 
# cas non binaire pour que la classe fonctionne sur y, il faut que le vecteur y soit onehot encodé
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
y_test_2d = y_test.reshape(len(y_test), 1)
y_test_multi = onehot_encoder.fit_transform(y_test_2d)
from sklearn.metrics import roc_auc_score
# weighted : Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label).
roc_auc_score(y_test_multi,model.predict_proba(X_test),average='weighted')

# la courbe ROC on ne l'affiche pas car il faut choisir une modalité vs autres

  
import matplotlib.pyplot as plt
import numpy as np
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    # Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, model.predict(X_test))
np.set_printoptions(precision=2)

import itertools
# Plot non-normalized confusion matrix
class_names = ['setosa','versicolor','virginica']

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()