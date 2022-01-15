
# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler 
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage 
from matplotlib import pyplot as plt

from sklearn.cluster import AgglomerativeClustering

customer_data = pd.read_csv('shopping_data.csv')

#This will return (200, 5) which means that the dataset 
# contains 200 records and 5 attributes.
print(customer_data.shape)

#To see the dataset structure, 
# execute the head() function of the data frame:
print(customer_data.head())

data = customer_data.iloc[:, 3:5].values

print(data)

# Figure 1
labels = range(1, 201)
plt.figure(figsize=(10, 7)) 
plt.subplots_adjust(bottom=0.1) 
plt.scatter(data[:,0],data[:,1], label='True Position') 
for label, x, y in zip(labels, data[:, 0], data[:, 1]):
    plt.annotate(label,xy=(x, y),xytext=(-3, 3),
    textcoords='offset points', ha='right',va='bottom') 
plt.show()

# Figure 2
linked = linkage(data, 'ward')
labelList = range(1, 201)
plt.figure(figsize=(10, 7)) 
dendrogram(linked, orientation='top',
    labels=labelList, distance_sort='descending', 
    show_leaf_counts=True)
plt.show()

# Figure 3 
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward') 
cluster.fit_predict(data)

plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow') 
plt.show()