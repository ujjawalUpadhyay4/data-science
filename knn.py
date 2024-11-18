import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_excel('knn_data.xlsx')
# print(df)
x = df[['weight(x)','height(y)']].values
y = df['class'].values

# training,testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
df = np.array([[57,170]])
y_pred = knn.predict(df)
print(y_pred)
