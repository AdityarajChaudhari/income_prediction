import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv(r'D:\Datasets\hiring.csv')
data.head()

#data.shape
#data.info()

data['experience'].fillna(0,inplace=True)
data['test_score(out of 10)'].fillna(data['test_score(out of 10)'].mean(),inplace=True)

def convert(word):
    word_to_num = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_to_num[word]
data['experience'] = data['experience'].apply(lambda x : convert(x))

x = data.iloc[:,:3]
y = data.iloc[:,-1]

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)

pickle.dump(lr,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,8,5]]))