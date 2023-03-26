#!/usr/bin/env python
# coding: utf-8

# # NLP BASED TEXT CLUSTERING

# # Importing required packages

# In[1]:


import numpy as np 
import pandas as pd 
import string as st
import matplotlib.pyplot as plt
import re
import nltk
from nltk import PorterStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering 
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud


# # Reading the data

# In[2]:


with open('nlp_data.txt') as f:
    data = f.readlines()
    


# In[3]:


# Converting data into data frame
df = pd.DataFrame(data)
df


# # Data Pre processing

# In[4]:


#changing the column name

df['corpus']=df[0]
df.head()


# In[5]:


# Dropping unwanted columns
df.drop(df.columns[0],axis=1,inplace=True)


# In[6]:


# Removing all punctuations from the text

def remove_punct(text):
    return ("".join([ch for ch in text if ch not in st.punctuation]))
df['new'] = df['corpus'].apply(lambda x: remove_punct(x))
# Removing special characters
df['new'] = df['new'].replace({'§ ': ''},regex=True)
#Converting text to lowercase characters
df['new']  = df['new'].apply(lambda x: x.lower())
#Removing space,newline,tab
df['new']  = df['new'].apply(lambda x: re.sub(r'\s',' ',x))
#Removing any character which does not match to letter,digit or underscore
df['new']  = df['new'].apply(lambda x: re.sub(r'^\W+|\W+$',' ',x))
#Remove numbers
df['new'] = df['new'].str.replace('\d+','')



# In[7]:


df


# In[8]:


# Dropping rows with nan values
df['new'].replace(' ',np.nan,inplace=True)


# In[9]:


df = df.dropna()


# In[10]:


# Rearranging the index
df.index = np.arange(1,len(df) + 1)
df


# In[11]:


# Tokenizing the data

def tokenize(text):
    text = re.split('\s+' ,text)
    return [x.lower() for x in text]


# In[12]:


df['tokens'] = df['new'].apply(lambda msg : tokenize(msg))
df


# In[13]:


#Removing stop words
def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]


# In[14]:


df['clean_tokens'] = df['tokens'].apply(lambda x : remove_stopwords(x))
df


# # Stemming
# 
# Stemming is the process of reduction of a word into its root or stem word. The word affixes are removed leaving behind only the root form.
# 
# 

# In[15]:


# Applying stemming 
def stem(text):
    word_net = PorterStemmer()
    return [word_net.stem(word) for word in text]


# In[16]:


df['stem_words'] = df['clean_tokens'].apply(lambda x : stem(x))
df


# # Lemmatization
# 
# Lemmatization is a method for combining various inflected forms of words into a single root form with the same meaning. It's comparable to stemming because it results in a stripped-down word with dictionary meaning.
# 

# In[17]:


# Applying lemmatization
def lemmatize(text):
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]


# In[18]:


df['lemma_words'] = df['clean_tokens'].apply(lambda x : lemmatize(x))
df


# In[19]:


# Creating new text with stem words for input data beofre vectorization

def return_sentences(tokens):
    return " ".join([word for word in tokens])


# In[20]:


df['new_text'] = df['stem_words'].apply(lambda x : return_sentences(x))
df


# In[21]:


# Creating new data frame with new text
new_data = pd.DataFrame(df['new_text'])
new_data


# # TF - IDF
# 
# TF(Term Frequency) measures how often a term occurs in a given dataset
# 
# IDF(Inverse document frequency) measures the importance of the term across a corpus.

# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[23]:


tfidf = TfidfVectorizer()
tfidf_vect = tfidf.fit_transform(df['new_text'])


# In[24]:


tfidf_vect.shape


# In[25]:


#Changing to data frame

dense = tfidf_vect.todense()
denselist = dense.tolist()
data = pd.DataFrame(denselist,columns=tfidf.get_feature_names())


# In[26]:


data


# # Count Vectorization
# 
# It is used to convert a collection of text documents to a vector of term/token counts. It also enables the pre-processing of text data prior to generating the vector representation. 

# In[27]:


from sklearn.feature_extraction.text import CountVectorizer


# In[28]:


count_vectorizer = CountVectorizer()
count_vect = count_vectorizer.fit_transform(df['new_text'])
count_vect.shape


# In[29]:


#Changing to data frame

dense = count_vect.todense()
denselist = dense.tolist()
data2 = pd.DataFrame(denselist,columns=count_vectorizer.get_feature_names())


# In[30]:


data2


# # PCA
# 
# Principal Component Analysis (PCA) is a linear dimensionality reduction technique that can be utilized for extracting information from a high-dimensional space by projecting it into a lower-dimensional sub-space. It tries to preserve the essential parts that have more variation of the data and remove the non-essential parts with fewer variation.
# 
# It is an Unsupervised dimensionality reduction technique where you can cluster the similar data points based on the feature correlation.

# In[31]:


from sklearn.decomposition import PCA


# In[32]:


pca = PCA(.95)


# In[33]:


pca.fit(data)


# In[34]:


pca1 = pca.fit_transform(data)


# In[35]:


pca.n_components_


# In[36]:


#Apply the transformation and convert the result into a DataFrame.

columns = ['pca_%i' % i for i in range(483)]
df_pca = pd.DataFrame(pca.transform(data), columns=columns, index=data.index)
df_pca.head()


# PCA ON COUNT VECTORIZER DATA

# In[37]:


pca = PCA(.95)


# In[38]:


pca.fit(data2)


# In[39]:


pca.n_components_


# In[40]:


pca2 = pca.fit_transform(data2)


# In[41]:


#Apply the transformation and convert the result into a DataFrame.

columns = ['pca_%i' % i for i in range(268)]
df_pca2 = pd.DataFrame(pca.transform(data2), columns=columns, index=data2.index)
df_pca2.head()


# # DETERMINING NUMBER OF CLUSTERS 'K'
# 
# K MEANS

# In[42]:


# determining the optimal number of clusters using an elbow pot on tf-idf pca data
wcss = []
k = list(range(2,10))
for i in k :
    kmeans_cluster = KMeans(n_clusters=i)
    kmeans_cluster.fit(df_pca)
    wcss.append(kmeans_cluster.inertia_)


# In[43]:


plt.figure(figsize = (10,8))
plt.plot(k,wcss,'go--')
plt.xlabel('Number of centroids')
plt.ylabel('Within cluster sum-of-squares')
plt.title('Elbow Plot of tf-idf pca data')


# In[44]:


# determining the optimal number of clusters using an elbow pot on count vectorized pca data
wcss = []
k = list(range(2,10))
for i in k :
    kmeans_cluster = KMeans(n_clusters=i)
    kmeans_cluster.fit(df_pca2)
    wcss.append(kmeans_cluster.inertia_)


# In[45]:


plt.figure(figsize = (10,8))
plt.plot(k,wcss,'go--')
plt.xlabel('Number of centroids')
plt.ylabel('Within cluster sum-of-squares')
plt.title('Elbow Plot of count vectorised pca data')


# In[46]:


# determining the optimal number of clusters using an elbow pot on orginal vectorised  data
wcss = []
k = list(range(2,10))
for i in k :
    kmeans_cluster = KMeans(n_clusters=i)
    kmeans_cluster.fit(data)
    wcss.append(kmeans_cluster.inertia_)


# In[47]:


plt.figure(figsize = (10,8))
plt.plot(k,wcss,'go--')
plt.xlabel('Number of centroids')
plt.ylabel('Within cluster sum-of-squares')
plt.title('Elbow Plot on orginal data set')


# From the elbow plot, we can see that were are not getting the optimal number of clusters used to cluster our data.
# 
# Hence, we check with another method using depth difference to determine the optimal number of clusters

# # Depth difference
# 
# The DeD method estimates the k parameter before actual clustering is constructed. We define
# the depth within clusters, depth between clusters, and depth diference to finalize the optimal value of k, which is an input
# value for the clustering algorithm. 

# In[48]:


from sklearn.neighbors import DistanceMetric


# In[49]:


# Defining a function to calculate Mahalanobis depth 

def mahalanobis(data=None,cov=None):
    x=data[data.columns].values
    covariance=np.linalg.inv(np.cov(x.T))
    dist=DistanceMetric.get_metric('mahalanobis',VI=covariance) #inverse of cov inbuilt fn
    mahal_dist=dist.pairwise(x)
    m_depth=1/(1+mahal_dist)
    depth=np.median(m_depth,axis=0)
    return(depth)


# In[50]:


# Calling the function on tf-idf PCA transformed data
pca_Mahalanobis = mahalanobis(data=df_pca)


# In[51]:


pca_Mahalanobis


# In[52]:


# Converting into dataframe
pca_mahal = pd.DataFrame(pca_Mahalanobis)


# In[53]:


pca_mahal


# In[54]:


#DeD Method

DM=pca_mahal.max()
diff=abs((pca_mahal-DM))
avg_diff=diff.mean()
n=df_pca.shape[0]



depth_difference=[]
for k in range(2,10):
    range_ = n//k
    start = 0
    end = 0
    init = 0
    for j in range(1, k-1):
        print(j)
        start = end + 1
        end = start + range_ - 1
        Di = pca_mahal.iloc[start:end]
        DepM = Di[0].max()
        avg = np.mean((abs((Di-DepM))))
        init = init+avg
       
    #Depth within clusters
    DW = init/k
    
    
    # Depth between cluster
    DB = avg_diff-DW
    
    
    Ded = DW - DB
    
    
    
    depth_difference.append(Ded)
    
    
print(depth_difference)


# In[55]:


plt.figure(figsize=(7,7))
plt.plot(range(2,10),depth_difference,marker='o')
plt.title('Depth Difference Plot')
plt.xlabel('Number of Clusters')
plt.ylabel('Depth Difference')
plt.show()


# From this plot we get 8 to be the optimal number of clusters.

# # Model Building
# 

# # K Means Clustering

# In[107]:


# Fitting the optimal number of clusters on data
kmeans_optimal = KMeans(n_clusters=8)
kmeans_optimal.fit(pca1)


# In[108]:


y_pred = kmeans_optimal.predict(pca1)
print(y_pred)


# In[109]:


pca_copy = df_pca.copy()
pca_copy


# In[110]:


pca_copy['Cluster'] = y_pred+1
pca_copy


# In[76]:


from collections import Counter


# In[77]:


count = Counter(kmeans_optimal.labels_)
print(count)


# In[60]:


from sklearn.metrics import silhouette_score


# In[61]:


print(f'Silhouette Score(n=8): {silhouette_score(pca_copy, y_pred)}')


# In[62]:


#Plot of clusters
plt.figure(figsize=(10,8))
plt.scatter(pca1[y_pred == 0, 0], pca1[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')  
plt.scatter(pca1[y_pred == 1, 0], pca1[y_pred == 1, 1], s = 100, c = 'black', label = 'Cluster 2')  
plt.scatter(pca1[y_pred== 2, 0], pca1[y_pred == 2, 1], s = 100, c = 'cyan', label = 'Cluster 3')  
plt.scatter(pca1[y_pred == 3, 0], pca1[y_pred == 3, 1], s = 100, c = 'magenta', label = 'Cluster 4')  
plt.scatter(pca1[y_pred == 4, 0], pca1[y_pred == 4, 1], s = 100, c = 'yellow', label = 'Cluster 5')  
plt.scatter(pca1[y_pred == 5, 0], pca1[y_pred == 5, 1], s = 100, c = 'green', label = 'Cluster 6') 
plt.scatter(pca1[y_pred == 6, 0], pca1[y_pred == 6, 1], s = 100, c = 'blue', label = 'Cluster 7') 
plt.scatter(pca1[y_pred == 7, 0], pca1[y_pred == 7, 1], s = 100, c = 'pink', label = 'Cluster 8') 

plt.title('K Means Clustering Model')
plt.legend()
plt.show()


# In[65]:


# K Means on count vectorised data after pca with k=8

# Fitting the optimal number of clusters on data
kmeans_optimal = KMeans(n_clusters=8,random_state=42)
kmeans_optimal.fit(pca2)


# In[66]:


cv_pred = kmeans_optimal.predict(pca2)
print(cv_pred)


# In[68]:


pca_cv = df_pca2.copy()


# In[72]:


pca_cv['cluster']= cv_pred + 1


# In[73]:


pca_cv


# In[74]:


#Plot of clusters
plt.figure(figsize=(10,8))
plt.scatter(pca2[cv_pred == 0, 0], pca2[cv_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')  
plt.scatter(pca2[cv_pred == 1, 0], pca2[cv_pred == 1, 1], s = 100, c = 'black', label = 'Cluster 2')  
plt.scatter(pca2[cv_pred== 2, 0], pca2[cv_pred == 2, 1], s = 100, c = 'cyan', label = 'Cluster 3')  
plt.scatter(pca2[cv_pred == 3, 0], pca2[cv_pred == 3, 1], s = 100, c = 'magenta', label = 'Cluster 4')  
plt.scatter(pca2[cv_pred == 4, 0], pca2[cv_pred == 4, 1], s = 100, c = 'yellow', label = 'Cluster 5')  
plt.scatter(pca2[cv_pred == 5, 0], pca2[cv_pred == 5, 1], s = 100, c = 'green', label = 'Cluster 6') 
plt.scatter(pca2[cv_pred == 6, 0], pca2[cv_pred == 6, 1], s = 100, c = 'blue', label = 'Cluster 7') 
plt.scatter(pca2[cv_pred == 7, 0], pca2[cv_pred == 7, 1], s = 100, c = 'pink', label = 'Cluster 8') 

plt.title('K Means Clustering Model')
plt.legend()
plt.show()


# We choose the tf-idf vectorized data over the count vectorized data since the clustering results were more accurate and it worked more efficiently whereas overlapping of the different clusters was largely observed in the count vectorized data. 
# 
# 

# # Hierarchal Clustering

# In[97]:


import numpy as np
import seaborn as sns
from matplotlib import  pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering


# In[98]:


# Fitting hierarchical clustering to the dataset

from sklearn.cluster import AgglomerativeClustering 
ac = AgglomerativeClustering(n_clusters = 8, affinity = 'euclidean', linkage ='ward')

# fitting the hierarchical clustering algorithm  to dataset  while creating the clusters 
y_ac=ac.fit_predict(pca1)


# In[99]:


pca_copy2 = df_pca.copy()
pca_copy2['ClusterH'] = y_ac + 1
pca_copy2.head(10)


# In[70]:


#Visualizing the clusters. This code is similar to k-means visualization code.
#We only replace the y_kmeans vector name to y_ac for the hierarchical clustering 
plt.figure(figsize=(10,7))
plt.scatter(pca1[y_ac==0, 0], pca1[y_ac==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(pca1[y_ac==1, 0], pca1[y_ac==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(pca1[y_ac==2, 0], pca1[y_ac==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(pca1[y_ac==3, 0], pca1[y_ac==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(pca1[y_ac==4, 0], pca1[y_ac==4, 1], s=100, c='magenta', label ='Cluster 5')
plt.scatter(pca1[y_ac== 5, 0],pca1[y_ac== 5, 1], s = 100, c = 'green', label = 'Cluster 6') 
plt.scatter(pca1[y_ac== 6, 0],pca1[y_ac== 6, 1], s = 100, c = 'blue', label = 'Cluster 7') 
plt.scatter(pca1[y_ac== 7, 0],pca1[y_ac== 7, 1], s = 100, c = 'pink', label = 'Cluster 8') 
plt.legend()

plt.title('Hierarchical Clustering Model')
plt.show()


# # Spectral Clustering

# In[101]:


from sklearn.cluster import SpectralClustering 

spectral_model_rbf = SpectralClustering(n_clusters = 8, affinity ='rbf')

y_sc = spectral_model_rbf.fit_predict(pca1)


# In[102]:


pca_copy3 = df_pca.copy()
pca_copy3['ClusterS'] = y_sc + 1
pca_copy3.head(10)


# In[74]:


plt.figure(figsize=(10,7))
plt.scatter(pca1[y_sc==0, 0], pca1[y_sc==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(pca1[y_sc==1, 0], pca1[y_sc==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(pca1[y_sc==2, 0], pca1[y_sc==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(pca1[y_sc==3, 0], pca1[y_sc==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(pca1[y_sc==4, 0], pca1[y_sc==4, 1], s=100, c='magenta', label ='Cluster 5')
plt.scatter(pca1[y_sc== 5, 0],pca1[y_sc== 5, 1], s = 100, c = 'green', label = 'Cluster 6') 
plt.scatter(pca1[y_sc== 6, 0],pca1[y_sc== 6, 1], s = 100, c = 'blue', label = 'Cluster 7') 
plt.scatter(pca1[y_sc== 7, 0],pca1[y_sc== 7, 1], s = 100, c = 'pink', label = 'Cluster 8') 
plt.legend()
plt.rcParams
plt.title('Spectral Clustering Model')
plt.show()


# # Word Cloud

# In[78]:


from wordcloud import WordCloud


# In[79]:


t1=df['new_text']


# In[80]:


word_cloud = WordCloud(collocations = False, background_color = 'white').generate(str(t1))


# In[81]:


plt.imshow(word_cloud)
plt.axis("off")
plt.title("Word Cloud of Pre-processed data")


# # Evaluation

# # Calinski Harabasz (CH) Index

# In[94]:


from sklearn.metrics import calinski_harabasz_score


# In[125]:


ch_scoreK = calinski_harabasz_score(pca_copy, y_pred)
print(ch_scoreK)


# In[112]:


ch_scoreH = calinski_harabasz_score(pca_copy2, y_ac)
print(ch_scoreH)


# In[113]:


ch_scoreS = calinski_harabasz_score(pca_copy3, y_sc)
print(ch_scoreS)


# # David Bouldin Index (DB)

# In[114]:


from sklearn.metrics import davies_bouldin_score


# In[118]:


db_indexK = davies_bouldin_score(pca1,y_pred)
print(db_indexK)


# In[119]:


db_indexH = davies_bouldin_score(pca1,y_ac)
print(db_indexH)


# In[120]:


db_indexS = davies_bouldin_score(pca1,y_sc)
print(db_indexS)


# Higher the CB index, the better the model. However,for DB Index, the lower the average similarity is,the better the clusters are divided and the better the clustering result is. Hence , we take Spectral Clustering to be the most efficent model.

# # Word Cloud of each cluster in Spectral Clustering

# In[86]:


data3 = pd.DataFrame(df['new_text'])


# In[124]:


data3['ClusterS']= y_sc + 1


# In[122]:


data3


# In[93]:


##Word cloud for each cluster

for i in range(1, 9):
    
    data1 = data3[data3['ClusterS']==i]
    data1 = data1.drop(['ClusterS'], axis = 1)
    data1 = " ".join(i for i in data1['new_text'])
    wordcloud = WordCloud().generate(data1)                      
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


# # Conclusion
# 
# The religious text data set was initially preprocessed to improve data quality using several NLP techniques. To map the pre-processed textual data to a numerical representation,tf-idf vectorization gave the most favorable results. Based on these numerical representations the data similarity could be determined, after reducing its dimension using PCA, we were able to obtain the optimal number of clusters through the Ded method which overpowered the K means elbow plot method. Using validity indices we were able to evaluate the clustering algorithms performed. Using Spectral clustering algorithm, we were able to effectively partition the text data to its eight labels of religious books.
# 
