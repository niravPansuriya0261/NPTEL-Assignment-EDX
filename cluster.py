import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel('College1_Data.xlsx',parse_cols='J,K,L,M') #change column name for change course
df=df[1:]

X=df.values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=900,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('elbow method')
plt.xlabel('number of clusters')    
plt.ylabel('WCSS')
plt.show()

    
    
kmeans=KMeans(n_clusters=2,init='k-means++',max_iter=900,n_init=16,random_state=0)
y_kmenas=kmeans.fit_predict(X)

plt.scatter(X[y_kmenas==0,0],X[y_kmenas==0,1],c='blue',label='pass')
plt.scatter(X[y_kmenas==1,0],X[y_kmenas==1,1],c='red',label='fail')
plt.legend()
plt.show()

count_0=(y_kmenas==0).sum()

percent = (count_0/480)*100
