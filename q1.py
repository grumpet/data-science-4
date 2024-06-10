import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

le = LabelEncoder()


df = pd.read_csv('votersdata.csv')
print(df.info())
#sex cataegorical
#age numerical
#salary numerical
#volunteering numerical ???
#passtime categorical
#status categorical
#vote categorical
#2a
catagorical = ['sex','passtime','status']

for var in catagorical:
    ct = pd.crosstab(df[var], df['vote'])
    ct.plot(kind='bar', stacked=True)
    plt.title(var)
    plt.show()
#2b
numerical = ['age','salary','volunteering']
for var in numerical:
    sns.boxplot(x='vote', y=var, data=df)
    plt.title(var)
    plt.show()
#3
df['sex'] = le.fit_transform(df['sex'])
df['passtime'].fillna('unknown', inplace=True)
df['passtime'] = le.fit_transform(df['passtime'])
df['status'] = le.fit_transform(df['status'])
df['vote'] = le.fit_transform(df['vote'])


#1
np.random.seed(123)

x=df.drop('vote',axis=1)
y = df['vote']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

clf = DecisionTreeClassifier(random_state=123)
clf.fit(x_train, y_train)


