# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 10:58:35 2018

@author: dgr
"""


# ACP sur les var numériques
# l'ACP doit être appliqué sur des données standardisées
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# MEDV est plus une variable qu'on veut projetter
X = housing.drop(["CHAS","MEDV",'Valeur_Immo'],axis=1)
# on écrase et on standardise
X = StandardScaler().fit_transform(housing.drop(["CHAS","MEDV",'Valeur_Immo'],axis=1))

X.mean(axis=0)

X.std(axis=0)

pca_housing = PCA(n_components=X.shape[1]).fit(X)
CP_housing = pca_housing.transform(X)

pca_housing.explained_variance_ratio_ # 3 composantes expliquent 72% de la variance des données
np.cumsum(np.round(pca_housing.explained_variance_ratio_, decimals=4)*100)
plt.plot(np.cumsum(np.round(pca_housing.explained_variance_ratio_, decimals=4)*100))
# les coordonnées des individus sur les deux axes
pd.DataFrame(CP_housing[:,0:2], columns = ['CP1', 'CP2']).head()

# examine l'objet
pca_housing.__dict__
from math import sqrt 

# valeur propre λi=si**2/(n−1)
list(map(lambda x:(x**2)/149,np.array(pca_housing.singular_values_)))
import seaborn as sns
targets = housing.Valeur_Immo.unique()


finalDf = pd.concat([pd.DataFrame(CP_housing[:,0:3], columns = ['CP1', 'CP2','CP3']), housing[["CHAS","MEDV",'Valeur_Immo']]], axis = 1)
sns.pairplot(x_vars='CP1', y_vars='CP2', data=finalDf, hue="Valeur_Immo", size=5)

sns.pairplot(x_vars='CP1', y_vars='CP3', data=finalDf, hue="Valeur_Immo", size=5)
# Les 3 catégories de valeur immobilière sont assez corrélées avec l'axe 1 et 3

#CP1 correle + avec
#CRIM
#INDUS
#NOX
#AGE
#RAD
#LSTAT
#
#et - avec
#ZN
#DIS
#TAX
#
#CP2 +
#CRIM ZN
#RAD
#TAX
#
#- black
#age
#
#CP3
#RM +
#et ptration - 
#
#à noter que MEDV est corree - avec CP1
#et + avec CP3 et orthogonal à CP2

del finalDf, stats
# corrélations etre les axes et les variables
#np.corrcoef(matNum,rowvar = False)
# 7 valeurs propres >1
analyse = pd.concat([pd.DataFrame(CP_housing[:,0:3], columns = ['CP1', 'CP2','CP3']), housing[["CRIM","ZN","INDUS","NOX","RM","AGE","DIS","RAD","TAX", "PTRATIO","BlackPop","LSTAT","MEDV"]]], axis = 1)
corr=analyse.corr().loc[["CRIM","ZN","INDUS","NOX","RM","AGE","DIS","RAD","TAX", "PTRATIO","BlackPop","LSTAT","MEDV"],["CP1","CP2","CP3"]]
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.index.values)

del corr
###########################################################################################
# Clustering
###########################################################################################
from sklearn.cluster import KMeans
from sklearn import metrics
#Cet algorithme étant aléatoire, on initialise la graine (avec random_state). 
# On choisit le classifieur à utiliser avec notamment le nombre de clusters à utiliser
k_means = KMeans(n_clusters = 3, random_state = 2016)
k_means.fit(CP_housing[:,0:3])
k_means.inertia_ #Sum of squared distances of samples to their closest cluster center.

# somme des distances de chacun à son barycentre gagnant / n

from scipy.spatial.distance import cdist
# sum(np.min(cdist(CP_housing[:,0:3], k_means.cluster_centers_, 'euclidean'), axis=1)) / CP_housing[:,0:3].shape[0]

# On regarde le cluster qui est assigné à chaque observation
pd.DataFrame(k_means.labels_, columns = ['Cluster']).head()
pd.DataFrame(k_means.labels_)[0].value_counts()
pd.DataFrame(k_means.labels_)[0].value_counts().apply(lambda x:x/len(hac.labels_))

#Pour optimiser le nombre de clusters, on fait tourner l’algorithme des kmeans pour plusieurs tailles de classification 
#et on les compare au sens de la statistique de Calinski-Harabasz 
#et du coefficient de Silhouette (un score élevé indique des clusters bien séparés).
choice = pd.DataFrame(columns = ['clusters','calinski','silhouette'])
# 3 à 8 clusters

## dist eucl entre chaque point et chaque barycentre
#dist=cdist(CP_housing[:,0:3], k_means.cluster_centers_, 'euclidean')
## dist entre chaque point et le barycentre dont il est le plus proche
#distm=np.min(cdist(CP_housing[:,0:3], k_means.cluster_centers_, 'euclidean'), axis=1)
## somme toutes les distances pour 
#sum(np.min(cdist(CP_housing[:,0:3], k_means.cluster_centers_, 'euclidean'), axis=1)) / CP_housing[:,0:3].shape[0]


distorsions=[]

for i in range(3,9) :
    k_means = KMeans(n_clusters = i, random_state = 2016)
    k_means.fit(CP_housing[:,0:3])
    choice.loc[i, 'clusters'] = i
    distorsions.append(sum(np.min(cdist(CP_housing[:,0:3], k_means.cluster_centers_, 'euclidean'), axis=1)) / CP_housing[:,0:3].shape[0])

    choice.loc[i, 'calinski'] = metrics.calinski_harabaz_score(CP_housing[:,0:3], k_means.labels_)
    choice.loc[i, 'silhouette'] = metrics.silhouette_score(CP_housing[:,0:3], k_means.labels_, metric='euclidean')
print(choice) 
# Calinski score maximisé avec 3clusters, Silhouette avec 3 -> ...

# On choisit 6 clusters et on sauvegarde les résultats dans la matrice cluster
k_means = KMeans(n_clusters = 3, random_state = 2016).fit(CP_housing[:,0:3])
typologie = pd.concat([analyse, pd.DataFrame(k_means.labels_, columns = ['K-means'])], axis = 1)

# On peut construire une représentation des points dans le plan principal des variables 
#(acp sur les 4 var, et on récupère les 2 premieres composantes)

sns.lmplot('CP1','CP3', data = typologie, hue = 'K-means', fit_reg=False)


###########################################################################################
# Clustering CAH
###########################################################################################

#La suite d’instructions calcule toutes les distances 2 à 2 puis calcule et dessine le dendrogramme : 
#    choisir la classification voulue, en utilisant la librarie SciPy.
# Une fois le nombre de cluster choisi, on peut lancer le clustering avec la librarie Scikit-learn.
## On utilise la librarie Scipy pour construire le dendogramme et nous permettre de choisir le nombre de clusters
from scipy import cluster
#scipy.cluster.hierarchy.linkage
# Linkage matrix
Z = cluster.hierarchy.linkage(CP_housing[:,0:3], method='ward', metric='euclidean')

# Dendogramme
plt.title('Hierarchical Clustering Dendrogram (truncated)', fontsize=18)
plt.xlabel('sample index', fontsize=16)
plt.ylabel('distance', fontsize=16)
# truncate mode permet d'afficher que le haut de l'arbre pour une profondeur de p split
# p n'est pas le nb de cluster
cluster.hierarchy.dendrogram(Z, p=4, truncate_mode = 'level', leaf_font_size=12,  leaf_rotation=90.,color_threshold =0) 
plt.show()

## On peut ensuite réaliser le clustering à l'aide de scikit-learn

# On choisit le classifieur à utiliser avec notamment le nombre de clusters à utiliser
# AgglomerativeClustering = CAH mais sans le dendogramme
from sklearn.cluster import AgglomerativeClustering
hac = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')

# On ajuste le modèle
hac.fit(CP_housing[:,0:3])

# On regarde le cluster qui est assigné à chaque observation
pd.DataFrame(hac.labels_,columns = ['Cluster']).head()
# comptage des clusters
pd.DataFrame(hac.labels_)[0].value_counts()
pd.DataFrame(hac.labels_)[0].value_counts().apply(lambda x:x/len(hac.labels_))


# On sauvegarde les résultats dans la matrice cluster
typologie = pd.concat([typologie, pd.DataFrame(hac.labels_, columns = ['HAC'])], axis = 1)

# On peut regarder la matrice de contingence pour voir les différences entre les classifications
pd.crosstab(typologie['HAC'], typologie['K-means'])
del typologie
###########################################################################################
# Clustering MIXTE
###########################################################################################

#La classification mixte permet de garder les qualités des méthodes de nuées dynamiques 
#et de la classification hiérarchique en contournant leurs inconvénients.
# On commence par une k-means pour diminuer la dimension du problème 
# puis on fait une classification hiérarchique sur les centroides.


# On fait une k-means avec beaucoup de clusters et on récupère les coordonnées des centroides

k_means_cent = KMeans(n_clusters = 25, random_state = 2016).fit(CP_housing[:,0:3])
# barycentres des 25 classes issues de la partition KM
centroides = k_means_cent.cluster_centers_

# 25 individus fictifs (barycentres)

# On trace le dendogramme 
# étape préliminaire avec la cluster.hierarchy (CAH): bénéfice de pouvoir représenter
Z = cluster.hierarchy.linkage(centroides, method='ward', metric='euclidean')
plt.figure()

plt.title('Hierarchical Clustering Dendrogram', fontsize=18)

plt.xlabel('sample index', fontsize=16)

plt.ylabel('distance', fontsize=16)
cluster.hierarchy.dendrogram(Z, leaf_font_size=12, leaf_rotation=90.) 

# etape partitionner par rapport au nombre = 3 ou 4
# On peut ensuite réaliser le clustering à l'aide de scikit-learn
hac_cent = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
hac_cent.fit(centroides)


# On récupère le cluster assigné lors du K-means puis lors du CAH pour avoir un cluster final pour chaque observation
kmeans_cent_df = pd.DataFrame(k_means_cent.labels_, columns = ['K-means_cent']).reset_index()
hac_cent_df = pd.DataFrame(hac_cent.labels_, columns = ['Typologie']).reset_index()
cluster_cent = pd.merge(kmeans_cent_df, hac_cent_df, left_on=['K-means_cent'], right_on=['index'], how='left')
cluster_cent = cluster_cent.drop(['K-means_cent','index_y'], axis =1)

cluster_cent.Typologie.value_counts()
cluster_cent.Typologie.value_counts().apply(lambda x:x/len(cluster_cent.Typologie))

typologie = pd.merge(left=housing, right=cluster_cent, how='inner',
		   left_index=True, right_index=True)

barycentres=typologie.groupby(['Typologie'])[["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX", "PTRATIO","BlackPop","LSTAT","MEDV"]].mean()

#Pour chaque cluster, on calcule la moyenne des variables quantitatives
# et la proportion des variables qualitatives. On peut ainsi voir en quels domaines les clusters se différencient. Ci-dessous le portrait robot de la classification mixte :

# Variables quanti : moyenne de la variable
# le index =false permet que le resultat reprenne bien les var en group by
stats=typologie.groupby(['Typologie'])[["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX", "PTRATIO","BlackPop","LSTAT","MEDV"]].mean()

# Variables quali : proportion de la variable dans chaque catégorie
pd.crosstab(typologie['Typologie'], typologie['CHAS'], normalize=True)
pd.crosstab(typologie['Typologie'], typologie['Valeur_Immo'], normalize='index')

#On peut également calculer les p-values associés à chaque variable quantitative pour chaque groupe pour savoir si la distribution de la variable est significativement différente entre le groupe A et l'ensemble des autres groupes. On calcule donc autant de p-values qu'il y a de modalités (p-value de la modalité 1 vs les autres modalités, puis p-value modalité 2 vs les autres etc., et ceci pour chaque variable).
import scipy.stats as sp
def portrait_robot_quanti(data, y, X):
    
    # Calcul des portraits robots en fonction de la variable cible et 
    # calcul du t-test pour savoir si le groupe cible est significativement différent de la population totale
    
    # Variables en entrée :
    # data = la base de données
    # y = le nom de la variable cible
    # X = la liste des noms des variables explicatives
    
    # Portraits robots 
    moy_tous = data[X].mean()
    moy_tous = pd.DataFrame(data[X].mean())
    moy_tous.rename(columns = {0:'All'}, inplace=True)
    
    moy_class = data.groupby([y], as_index = False)[X].mean()
    moy_class[y] = data[y].value_counts()
    moy_class.rename(columns= {y : 'Effectif'}, inplace=True)
    moy_class = moy_class.T

    typo = pd.concat([moy_class, moy_tous], axis=1)
    typo.loc['Effectif', 'All'] = data.shape[0]
    typo.rename(columns = dict([(i, 'y_'+ str(i)) for i in typo.columns.values]), inplace=True)
    
    # Pvalues
    t_test = pd.DataFrame(columns = data[y].unique().tolist() + ['VARIABLE']) 
    for i in data[y].unique().tolist() :
        for j in range(0, len(X)) :  
            t_test.loc[j,'VARIABLE'] = X[j]
            t_test.loc[j,i] = sp.ttest_ind(data[data[y] == i][X[j]], data[data[y] != i][X[j]])[1]   
    t_test.rename(columns = dict([(i, 'p-value_'+str(i)) for i in data[y].unique().tolist()]), inplace=True)
    
    # Jointure
    res = pd.merge(typo, t_test, left_index = True, right_on = 'VARIABLE', how='outer')
    res = res[['VARIABLE'] + res.columns.sort_values()[1:].tolist()]
    res = res.sort_values(by = 'VARIABLE')
    res = pd.concat([res[res['VARIABLE']=='Effectif'], res[res['VARIABLE']!='Effectif']],axis=0)
    
    return res


stats=portrait_robot_quanti(typologie, 'Typologie', ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX", "PTRATIO","BlackPop","LSTAT","MEDV"])

# la classe 1 : ventre mou (40% dela base)
# la classe 2 : villes proche de l'hypercentre
# la classe 3 : banlieue plus pauvres mais proches bassins demplo 
# la classe 4 : banlieues rÃ©sidentielles

###########################################################################################
# Réaffection
###########################################################################################

#Dans cette section, il s’agit de proposer des techniques d’affectation à une segmentation pré-existante et dont nous possédons un échantillon : nous avons des observations sur des individus dont nous connaissons le segment. On est ici dans un cadre de classification supervisée.
#On commence par séparer notre base en deux : une partie pour apprendre le lien entre les variables explicatives et la cible (la classe d’appartenance du client : ici la classification mixte), et une autre sur laquelle on évalue la qualité de la méthode d’affectation - principe de ne pas être juge et partie.
#On choisit la taille respective des bases - ici 75% pour l’apprentissage - , puis on tire 75% des lignes au hasard que l’on affecte à la base train, le reste étant affecté à test.

rateEch = 0.75
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(typologie[["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX", "PTRATIO","BlackPop","LSTAT"]],typologie.Typologie,train_size = np.round(len(typologie) * rateEch).astype(int), random_state = 2016)

#Plus proches voisins

#On cherche les plus proches voisins de chaque individu et on lui affecte la classe majoritaire parmi ses voisins. 
#Le paramètre de la méthode est le nombre K de voisins - ici K = 8.
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 8)
knn.fit(X_train,y_train)

# ech test
y_pred_test = knn.predict(X_test)


#La qualité de la méthode se lit sur la matrice de confusion. Cette méthode est utilisée sur les variables quantitatives.
pd.crosstab(y_pred_test, y_test,normalize='index')


from sklearn.metrics import confusion_matrix,accuracy_score
accuracy_score(y_test,y_pred_test)

confusion_matrix(y_test,y_pred_test)
list(map(lambda x:x/sum(x),confusion_matrix(y_test,y_pred_test)))
confusion_matrix(y_test,y_pred_test)

stat=pd.DataFrame(list(map(lambda x:x/sum(x),confusion_matrix(y_test,y_pred_test))))

# individu décrit sur ses variables originales
housing_new= pd.read_csv('boston_new.csv',index_col=False,sep=',',
names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX", "PTRATIO","BlackPop","LSTAT","MEDV"])

# prédiction : on compare l'individu à la base de connaissance
predictions=knn.predict(housing_new[["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX", "PTRATIO","BlackPop","LSTAT"]])
