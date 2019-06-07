# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 15:12:29 2018

@author: dgr
"""

# librairies dapp
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from fonctions_metrics import lift,CAP_table

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score,roc_auc_score,roc_curve,auc

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#############################################################################################
#  Rég logistique lasso L1
############################################################################################
## Regression logistique avec pénalité lasso et grid search
#### on cherche par CV le meilleur C (1/alpha) le coef de regularisation

param = [{"C":(0.001,0.01,0.1,1,10,100,1000)}]
logitnow = GridSearchCV (LogisticRegression(penalty = "l1"), param, cv = 4,n_jobs=1,scoring='roc_auc')
#logit = GridSearchCV (LogisticRegression(penalty = "l1",class_weight='auto'), param, cv = 4,n_jobs=4,scoring='f1')
Grid_lasso = logitnow.fit(X_train,y_train)
# CV = 4 , validation croisée en scindant l'app en 4 (folds)
# meilleure grille
Grid_lasso.best_params_
#{'C': 0.1}

#Grid_lasso.best_params_

# design du modele
model_LogitL1 =LogisticRegression(penalty = "l1",C=0.1)
# Apprentissage du modele
model_LogitL1.fit(X_train,y_train)
coef=list(model_LogitL1.coef_[0]) # c un array2d, 
len(coef) # il y a 20 
# combien d'élemnts non nuls ?
feature_0= list(map(lambda x: x==0.0, coef)) #print(feature_0)
feature_0.count(True) # 6
del feature_0

# PLus on durcit, plus de colonnes s'annulent (0)
# print(pd.DataFrame({'Coefficients': list(reg.coef_)}, list(X.columns.values)))


model='Score Lasso'

# Prediction des probabilités de 1 , array2d
probas_test = model_LogitL1.predict_proba(X_test)[:,1]
probas_train = model_LogitL1.predict_proba(X_train)[:,1]
# score: accuracy taux de bien classé global
model_LogitL1.score(X_test, y_test)
model_LogitL1.score(X_train,y_train)

#AUC
roc_auc_score(y_train,probas_train)
roc_auc_score(y_test,probas_test)
# AUC 0.82 sur test
# sur-apprentissage : pas de
# pourquoi : car lasso par déf supprime
from fonctions_metrics import lift
#compute lift at 10%#
lift(probas_train,X_train,y_train)
lift(probas_test,X_test,y_test)
# 5.08
#compute lift at 5%
lift(probas_train,X_train,y_train,p=5)
lift(probas_test,X_test,y_test,p=5)

#CAP_table(pd.Series(y_test),pd.Series(probas_test),5,8)
# 5.9
t = pd.concat([pd.Series(y_test), pd.Series(probas_test)], axis = 1).reset_index()[[1,2]]

# métriques (liste de dictionnaires)
metriques = [{'model':model,'AUC_test':round(roc_auc_score(y_test,probas_test),2),'lift at 5':lift(probas_test,X_test,y_test,p=5),'lift at 10':lift(probas_test,X_test,y_test,p=10)}]


#Ordonner par ordre décroissant du score
calcul_lift =pd.concat([pd.DataFrame(y_test),pd.DataFrame(probas_test)],axis=1)
calcul_lift.columns=['Churn',model]
calcul_lift = calcul_lift.sort_values([model], ascending=False)
#Cumulative values du nombre de customers en %
calcul_lift.loc[:,'customers'] = np.arange(0, 1, 1/float(len(calcul_lift)))
#Calcul du true positive rate
calcul_lift.loc[:,'responders'] = np.cumsum(calcul_lift['Churn']) / np.sum(calcul_lift['Churn'])
calcul_lift['lift'] = calcul_lift['responders'] / calcul_lift['customers'] 
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0, y_test.sum()/len(y_test)], [0, 1], color='navy', lw=2, linestyle=':', label='Courbe parfaite')
plt.plot([y_test.sum()/len(y_test), 1], [1, 1], color='navy', lw=2, linestyle=':')
plt.plot(calcul_lift['customers'], calcul_lift['responders'], label = model, lw=2)
plt.xlabel('% Population', fontsize=16)
plt.ylabel('% Positive Responses', fontsize=16)
plt.title('Cumulative gains chart', fontsize=18)
plt.legend(loc="lower right", fontsize=16)
plt.figure(figsize=(8, 6))
plt.plot(calcul_lift['customers'], calcul_lift['lift'], label = model, lw=2)

#roc_test = roc_curve(y_test, probas_test)
# on recupere dans des array les compostantes de la courbe roc le fpr et le tpr par threshold
fpr, tpr, thresholds = roc_curve(y_test, probas_test)
roc_auc = auc(fpr, tpr) # calcul AUC
plt.figure()

plt.plot(fpr, tpr, lw=1, label='ROC fold (AUC = %0.2f)' % roc_auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()



# mat de confusion à un cutoff donné
pred = (probas_test > 0.14).astype(int)
print(confusion_matrix(y_test, pred))
# parmi les vrais 1 82% sont bien prédits
print(pd.crosstab(y_test,pred).apply(lambda r: r/r.sum(), axis=1))
# parmi les predits 1 34% sont des vrais positifs
print(pd.crosstab(y_test,pred).apply(lambda r: r/r.sum(), axis=0))

target_names = ['Fidèles','Churners']

# métriques au cutoff donné
print(classification_report(y_test,pred, target_names=target_names))
# métriques au cutoff par défaut
print(classification_report(y_test,model_LogitL1.predict(X_test), target_names=target_names))
