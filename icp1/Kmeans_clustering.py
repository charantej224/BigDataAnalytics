import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import metrics

## Reading the dataframe from the CSV file
df = pd.read_csv('Data/Data.csv')

'''Data Cleaning
Step1: drop the unique values as it doesnt mean of significance when understanding co-relation.
Step : the target values are considered as Malign or Benign cancer which are replaced as numbers rather than characters.
'''
df = df.drop("id", axis=1)
target_dict = {
    "M": 0,
    "B": 1
}
df["diagnosis"] = df["diagnosis"].apply(lambda x: target_dict[x])

features = df.iloc[:, 1:31]
prediction = df.iloc[:, 0:1]

print(features.shape, prediction.shape)

inertia_list = []

'''
k-means clustering to see how given data behaves with number of clusters.
'''

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(features)
    inertia_list.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia_list)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.savefig("plots/kmeans-cluster-analysis.png")
plt.show()
#
scaler = preprocessing.StandardScaler()
scaler.fit(features)
X_scaled_array = scaler.transform(features)
X_scaled = pd.DataFrame(X_scaled_array, columns=features.columns)

km = KMeans(n_clusters=2)
km.fit(features)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(features)
score = metrics.silhouette_score(features, y_cluster_kmeans)
print(score)

km.fit(X_scaled)
y_cluster_kmeans = km.predict(X_scaled)
score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
print(score)

'''
with reducing dimensionality from many features into two to see how the data is scatterred. 
this also shows if using PCA to do the dimensionality reduction actually helps.
'''

pca = PCA(n_components=2)
x_pca = pca.fit_transform(X_scaled_array)
df2 = pd.DataFrame(data=x_pca, columns=["pc1", "pc2"])

km.fit(df2)
df2['results'] = km.predict(df2)
score = metrics.silhouette_score(features, y_cluster_kmeans)
print(score)
sns.FacetGrid(df2, hue="results", height=4).map(plt.scatter, "pc1", "pc2").add_legend()
plt.savefig("plots/principle_component_analysis.png")
plt.show()
