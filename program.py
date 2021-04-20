#NAME-->SHEELAJ BABU
#BRANCH-->CSE

import numpy as np
import csv
import pandas as pd
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
import scipy.stats.mstats as ms
with open("04cars.csv", 'r') as f:
    wines = list(csv.reader(f, delimiter=";"))
wines = np.array(wines[1:])
print(wines.shape)
print(wines[:1])
df=pd.read_csv("04cars.csv")
feature_names=list(df.columns)
myFeature=np.array(feature_names[8:])
print(myFeature.shape)
for val in myFeature:
    print(val)
X = df[1:]
X.shape
X.columns
X=X.drop(['Name','SportsCar?','SUV?','Wagon?','Minivan?','Pickup?','AllWheelDrive?','RearWheelDrive?'],axis=1)
X.dtypes
# convert * to 0 in columns
X['CityMPG'] = X['CityMPG'].str.replace('*', "0")
X['HighwayMPG'] = X['HighwayMPG'].str.replace('*', "0")
X['Weight(lbs)'] = X['Weight(lbs)'].str.replace('*', "0")
X['WheelBase(in)'] = X['WheelBase(in)'].str.replace('*', "0")
X['Length(in)'] = X['Length(in)'].str.replace('*', "0")
X['Width(in)'] = X['Width(in)'].str.replace('*', "0")
# change object and float datatypes to interger
X['CityMPG']=pd.to_numeric(X['CityMPG'])
X['HighwayMPG']=pd.to_numeric(X['HighwayMPG'])
X['Weight(lbs)']=pd.to_numeric(X['Weight(lbs)'])
X['WheelBase(in)']=pd.to_numeric(X['WheelBase(in)'])
X['Length(in)']=pd.to_numeric(X['Length(in)'])
X['Width(in)']=pd.to_numeric(X['Width(in)'])
X['EngineSize(L)'] = X['EngineSize(L)'].astype(int)
#Setup the factor analysis model
fa = FactorAnalysis(n_components=2)
# fit the data
zX = ms.zscore(X)
f = fa.fit(zX)
W = ms.zscore(np.transpose(f.components_))
Psi = f.noise_variance_
# We first visualize and interpret the W matrix
fig = plt.figure(figsize=[10,8])
for i in range(W.shape[0]):
    plt.plot([0,W[i,0]],[0,W[i,1]])
    plt.text(W[i, 0], W[i, 1], myFeature[i])
plt.show()
# Find latent scores for each data point using the latent factors
Z = f.fit_transform(zX)
fig = plt.figure(figsize=[12, 10])
scatter = plt.scatter(Z[:, 0], Z[:, 1])
plt.show()
# Here End of my code #
