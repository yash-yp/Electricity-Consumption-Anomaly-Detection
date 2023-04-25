import pandas as pd
from sklearn.preprocessing import MinMaxScaler

rawData = pd.read_csv('data.csv')
infoData = pd.DataFrame()
infoData['FLAG'] = rawData['FLAG']
infoData['CONS_NO'] = rawData['CONS_NO']
data = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

# duplicates drop
dropIndex = data[data.duplicated()].index  
data = data.drop(dropIndex, axis=0)
infoData = infoData.drop(dropIndex, axis=0)

# zero rows drop
zeroIndex = data[(data.sum(axis=1) == 0)].index  
data = data.drop(zeroIndex, axis=0)
infoData = infoData.drop(zeroIndex, axis=0)

# columns reindexing according to dates
data.columns = pd.to_datetime(data.columns)  
data = data.reindex(sorted(data.columns), axis=1)
cols = data.columns

# index sorting
data.reset_index(inplace=True, drop=True)  
infoData.reset_index(inplace=True, drop=True)

# filling NaN values
data = data.interpolate(method='linear', limit=2,  
                        limit_direction='both', axis=0).fillna(0)

# outliers treatment
for i in range(data.shape[0]):  
    m = data.loc[i].mean()
    st = data.loc[i].std()
    data.loc[i] = data.loc[i].mask(data.loc[i] > (m + 3 * st), other=m + 3 * st)

# preprocessed data without scaling
data.to_csv(r'visualization.csv', index=False, header=True)  

scale = MinMaxScaler()
scaled = scale.fit_transform(data.values.T).T
mData = pd.DataFrame(data=scaled, columns=data.columns)
print(mData)

# Back to initial format
preprData = pd.concat([infoData, mData], axis=1, sort=False)  
print(preprData)
preprData.to_csv(r'preprocessedR.csv', index=False, header=True)
