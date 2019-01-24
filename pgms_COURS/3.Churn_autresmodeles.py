# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 15:12:58 2018

@author: dgr
"""



#############################################################################################
#  4 - Rég logistique Elasticnet
#############################################################################################


#### Régression logistique elastic net : SGDClassifier : loss = log --> logistique
#### penalty = "elasticnet"
#### alpha entre 1 et 1000 , et l1_ratio entre 0 et 1
from sklearn.linear_model import SGDClassifier
#Defaults to ‘hinge’, which gives a linear SVM. The ‘log’ loss gives logistic regression, a probabilistic classifier. 
#l1_ratio : float
#The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Defaults to 0.15

param = [{"alpha":(0.001,0.01,0.025,0.05,0.1,0.2)} ]
sgdElasticAUCnow = GridSearchCV (SGDClassifier(loss="log", penalty="elasticnet",l1_ratio=0.25,class_weight=None), param, cv = 4,n_jobs=4,scoring='roc_auc')

cvelasticnet = sgdElasticAUCnow.fit(X_train,y_train)
cvelasticnet.best_params_
#☺alpha 0.05
elasticnet = SGDClassifier(loss="log", penalty="elasticnet", alpha=0.05,l1_ratio=0.25,class_weight=None)
# Apprentissage du modele
elasticnet.fit(X_train,y_train)
elasticnet.coef_
  #compute lift at 10%#
  
model='Score Elasticnet'
# Prediction des probabilités de 1 , array2d
probas_test = elasticnet.predict_proba(X_test)[:,1]
probas_train = elasticnet.predict_proba(X_train)[:,1]
  #AUC
roc_auc_score(y_train,probas_train)
roc_auc_score(y_test,probas_test)
# AUC 0.8318 sur test

#compute lift at 10%#
lift(probas_train,X_train,y_train)
lift(probas_test,X_test,y_test)
# 4.92
#compute lift at 5%
lift(probas_test,X_test,y_test,p=5)

#param = [{"alpha":(0.0005,0.001,0.0015,0.002,0.003,0.004,0.005),"l1_ratio":(0.05,0.1,0.15,0.2,0.25,0.3,0.4)}]


#param = [{"alpha":(0.001,0.01,0.025,0.05,0.1,0.2)} ]
#sgdElasticAUC = GridSearchCV (SGDClassifier(loss="log", penalty="elasticnet",l1_ratio=0.25,class_weight='balanced'), param, cv = 4,n_jobs=4,scoring='roc_auc')
#
#cvelasticnet = sgdElasticAUC.fit(X_train,y_train)
#cvelasticnet.best_params_
#elasticnet = SGDClassifier(loss="log", penalty="elasticnet", alpha=0.1,l1_ratio=0.1,class_weight='balanced')
## Apprentissage du modele
#elasticnet.fit(X_train,y_train)

#############################################################################################
#  5 - Random Forest
#############################################################################################

# Parametrage de la classe
# on met des param pour un modle avec une complexité importante
# on verra ainsi apparaitre du sur apprentissag car la métrique AUC est bien plus faible sur l'ech d'app / test
rf = RandomForestClassifier(n_estimators = 300,
                                       criterion = "gini",
                                       max_features = "sqrt",
                                       max_depth = 8,
                                       min_samples_split = 30,
                                       min_samples_leaf = 20,                                       
                                       max_leaf_nodes = 100,
                                       bootstrap = True,
                                       oob_score = True,
                                       n_jobs = -1, # coeurs
                                       random_state = 1234,
                                       class_weight="balanced"
                                       )
# Apprentissage du modele
rf.fit(X_train,y_train)

# Prediction des probabilités de 1 , array2d
probas_test = rf.predict_proba(X_test)[:,1]
probas_train = rf.predict_proba(X_train)[:,1]
  #AUC
roc_auc_score(y_train,probas_train)
# 0.96
roc_auc_score(y_test,probas_test)
# AUC 0.91 sur test

#compute lift at 10%#
lift(probas_train,X_train,y_train)
lift(probas_test,X_test,y_test)
# x 7.6
#compute lift at 5%
lift(probas_test,X_test,y_test,p=5)
# 9.7

# métriques (liste de dictionnaires)
metriques = metriques+[{'model':'rf','AUC_test':round(roc_auc_score(y_test,probas_test),2),'lift at 5':lift(probas_test,X_test,y_test,p=5),'lift at 10':lift(probas_test,X_test,y_test,p=10)}]

# conclusion  le modele RF est plus predictif mais moins confiance dans sa capacité à etre robuste à moyen terme


#param = [{"n_estimators":(50,100,200,500,1000),"criterion":["gini","entropy"],"max_depth":(2,4,6,8,10,12,16)}]
##param = [{"n_estimators":(10,50,100,200,500,1000),"criterion":["gini","entropy"]}]
#tune_rf = GridSearchCV (RandomForestClassifier(bootstrap = True,oob_score = True,n_jobs = -1,verbose = 0,min_samples_split = 20),
#                        param,cv = 4,n_jobs=-1,scoring='roc_auc')
#
# tune_rf.fit(X_train,y_train)
# tune_rf.best_params_
#     #{'criterion': 'entropy', 'max_depth': 12, 'n_estimators': 500}
#
#rf_best = RandomForestClassifier(bootstrap = True,oob_score = True,n_jobs = -1,verbose = 0,min_samples_split = 20,
#                                 criterion = "entropy",max_depth = 12,n_estimators = 500)
## Apprentissage du modele
#rf_best.fit(X_train,y_train)


#############################################################################################
#  7 - Boosting
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
from sklearn.ensemble import GradientBoostingClassifier
###########
########### sans randomisation (subsample=1.0 = on prend tt l'échantillon)
###########

gbt_noRand05=GradientBoostingClassifier(loss='deviance', learning_rate=0.05,
                           n_estimators=500,
                           subsample=1.0, min_samples_split=20, min_samples_leaf=10,
                           max_depth=4)
# Apprentissage du modele
gbt_noRand05.fit(X_train,y_train)

niter=500
iter = np.arange(niter) + 1
test_deviance = np.zeros((niter,), dtype=np.float64)
# staged_decision_functio : décision fonction à chaque iteration
for i, y_pred in enumerate(gbt_noRand05.staged_decision_function(X_test)):
    # clf.loss_ assumes that y_test[i] in {0, 1}
    test_deviance[i] = gbt_noRand05.loss_(y_test, y_pred)

plt.figure(figsize=(8, 6))
# Erreur sur le test (evolution deviance)
plt.plot(iter,test_deviance,label='Test',color='darkorange')
        # min vers 100 
# Erreur sur apprentissage (evolution deviance)
plt.plot(iter,gbt_noRand05.train_score_,label='Apprentissage',color='navy')    
# Diminution de l'erreur rapport modele precedant (par rapport au oob)
#plt.plot(iter,gbt_noRand05.oob_improvement_)
plt.legend(loc="upper right", fontsize=12)
# Prediction des probabilités de 1 , array2d
probas_test = gbt_noRand05.predict_proba(X_test)[:,1]
probas_train = gbt_noRand05.predict_proba(X_train)[:,1]
  #AUC
roc_auc_score(y_train,probas_train)
# 1
roc_auc_score(y_test,probas_test)
# AUC 0.91 sur test

#compute lift at 10%#
lift(probas_train,X_train,y_train)
lift(probas_test,X_test,y_test)
# x 7.6
#compute lift at 5%
lift(probas_test,X_test,y_test,p=5)
# 10.36

###########
########### algo avec randomisation subsample=0.5 et max_feature = racine(nb variable)
###########
gbt_Rand=GradientBoostingClassifier(loss='deviance', learning_rate=0.05,
                           n_estimators=500,
                           subsample=0.5, min_samples_split=30, min_samples_leaf=20,
                           min_weight_fraction_leaf=0.005,
                           max_depth=3,max_leaf_nodes=12,max_features="sqrt")
# Apprentissage du modele
gbt_Rand.fit(X_train,y_train)
niter=500
iter = np.arange(niter) + 1
test_deviance = np.zeros((niter,), dtype=np.float64)
# staged_decision_functio : décision fonction à chaque iteration
for i, y_pred in enumerate(gbt_Rand.staged_decision_function(X_test)):
    # clf.loss_ assumes that y_test[i] in {0, 1}
    test_deviance[i] = gbt_Rand.loss_(y_test, y_pred)

# negative cumulative sum of oob improvements (améliration de l'erreur)
#out-of-bag (OOB) estimates can be a useful heuristic to estimate the “optimal” number of boosting iterations. OOB estimates are almost identical to cross-validation estimates but they can be computed on-the-fly without the need for repeated model fitting. OOB estimates are only available for Stochastic Gradient Boosting (i.e. subsample < 1.0), the estimates are derived from the improvement in loss based on the examples not included in the bootstrap sample (the so-called out-of-bag examples). The OOB estimator is a pessimistic estimator of the true test loss, but remains a fairly good approximation for a small number of trees.
cumsum = -np.cumsum(gbt_Rand.oob_improvement_)
plt.figure(figsize=(8, 6))

plt.plot(iter,test_deviance,label='Test',color='darkorange')
plt.plot(iter,gbt_Rand.train_score_,label='Apprentissage',color='navy')    
plt.figure(figsize=(8, 6))
plt.plot(iter,cumsum,label='OOB loss')
plt.figure(figsize=(8, 6))

plt.plot(iter,gbt_Rand.oob_improvement_,label='amélioration loss sur OOB')
        # min vers 100 
# Erreur sur apprentissage (evolution deviance)
plt.plot(iter,gbt_noRand05.train_score_,label='Apprentissage',color='navy')    

        # min vers 100
     #   pred_at_eachstage=enumerate(gbt_Rand.staged_predict(X_testCR),100)
gbt_Rand=GradientBoostingClassifier(loss='deviance', learning_rate=0.05,
                           n_estimators=150,
                           subsample=0.5, min_samples_split=30, min_samples_leaf=20,
                           min_weight_fraction_leaf=0.005,
                           max_depth=3,max_leaf_nodes=12,max_features="sqrt")
# Apprentissage du modele
gbt_Rand.fit(X_train,y_train)


probas_test = gbt_Rand.predict_proba(X_test)[:,1]
probas_train = gbt_Rand.predict_proba(X_train)[:,1]
  #AUC
roc_auc_score(y_train,probas_train)
# 0,979
roc_auc_score(y_test,probas_test)
# AUC 0.89 sur test

#compute lift at 10%#
lift(probas_train,X_train,y_train)
lift(probas_test,X_test,y_test)
# x 7.3
#compute lift at 5%
lift(probas_test,X_test,y_test,p=5)
# 9.1
#############################################################################################
#  SVM
############################################################################################


param = [{"C":(0.001,0.01,0.1,0.5),"gamma":(0.01,0.1,10,100,1000,)}]
SVM_tune = GridSearchCV (SVC(kernel='rbf',probability = False,class_weight=None,cache_size=4000), param, cv = 4,n_jobs=-1,scoring='roc_auc')

SVM_tune.fit(X_train,y_train)
SVM_tune.best_params_

# 'C': 0.5, 'gamma': 0.1
svm=SVC(C=0.5,gamma=0.1, kernel='rbf',probability=True,class_weight=None,cache_size=2000)
svm.fit(X_train,y_train)
# métriques AUC
AUC_modele(svm.predict_proba(X_train)[:, 1],y_train)
AUC_modele(svm.predict_proba(X_test)[:, 1],y_test)
lift(svm.predict_proba(X_train),X_train,y_train)
lift(svm.predict_proba(X_test),X_test,y_test)