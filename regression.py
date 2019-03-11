import pandas as pd
import numpy as np

df=pd.read_excel('College1_Data.xlsx',parse_cols='F,G,H,I') #change column name for change course
df=df[1:]

X=df.iloc[:,:-1].values
Y=df.iloc[:,3].values
Y=Y.reshape(-1,1)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)


imputer_Y = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_Y = imputer_Y.fit(Y)
Y = imputer_Y.transform(Y)

Y=Y.reshape(-1,1)
X=X[:,0]


from sklearn.cross_validation import train_test_split
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=1/3,random_state=0)

X_Train=X_Train.reshape(-1,1)
Y_Train=Y_Train.reshape(-1,1)
X_Test=X_Test.reshape(-1,1)
Y_Test=Y_Test.reshape(-1,1)


from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()
linearRegression.fit(X_Train,Y_Train)

y_pred=linearRegression.predict(X_Test)

from sklearn.metrics import mean_squared_error

mse=mean_squared_error(Y_Test,y_pred)
mse=np.sqrt(mse)

import matplotlib.pyplot as plt

plt.scatter(X_Test,Y_Test,color='red')
plt.plot(X_Test,y_pred,color='blue')
plt.xlabel('% video access')    
plt.ylabel('marks')
plt.show()

mse
