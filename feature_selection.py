import pandas as pd
import numpy as np

df=pd.read_excel('College1_Data.xlsx',parse_cols='F,G,H,I')  #change column name for change course
df=df[1:]

X=df.iloc[:,:-1].values
Y=df.iloc[:,3].values

import statsmodels.formula.api as sm

X=np.append(arr=np.ones((480,1)).astype(int),values=X,axis=1)

X_OPT=X[:,[0,1,2,3]] #I have used back elimination method.
Y = Y.astype(float)
X_OPT = X_OPT.astype(float)
regressor_OLS=sm.OLS(endog=Y,exog=X_OPT).fit()
regressor_OLS.summary()

X=X[:,1]
