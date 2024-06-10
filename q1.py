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
print("Sex classes:", dict(zip(le.classes_, le.transform(le.classes_))))

df['passtime'].fillna('unknown', inplace=True)
df['passtime'] = le.fit_transform(df['passtime'])
print("Passtime classes:", dict(zip(le.classes_, le.transform(le.classes_))))

df['status'] = le.fit_transform(df['status'])
print("Status classes:", dict(zip(le.classes_, le.transform(le.classes_))))

df['vote'] = le.fit_transform(df['vote'])
print("Vote classes:", dict(zip(le.classes_, le.transform(le.classes_))))

#1
np.random.seed(123)

x=df.drop('vote',axis=1)
y = df['vote']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

clf = DecisionTreeClassifier(random_state=123)
clf.fit(x_train, y_train)


#6
#model evaluation confusion matrix
y_pred = clf.predict(x_test)

# Calculate Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Calculate Precision
precision = metrics.precision_score(y_test, y_pred, pos_label=0)
print(f'Precision: {precision}')

# Calculate Recall
recall = metrics.recall_score(y_test, y_pred, pos_label=0)
print(f'Recall: {recall}')






