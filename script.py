# Imports
import collections

import pickle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

# further imports go here

# Load
with open('train_set_top_5.pkl', 'rb') as f:
    train_set_top_5 = pickle.load(f)
train_set_top_5 = np.array(train_set_top_5)
print(train_set_top_5[:3])

with open('test_set_top_5.pkl', 'rb') as f:
    test_set_top_5 = pickle.load(f)
test_set_top_5 = np.array(test_set_top_5)

# Inspect Labels

labels = [el["cuisine"] for el in train_set_top_5]
s = sorted(collections.Counter(labels).items(), key=lambda x: x[1], reverse=True)
labels_unique = np.unique(labels)
print(labels_unique)
# print(labels_pd.unique())
# print(s)


# Preprocessing & Feature Engineering
# Write your preprocessing and feature engineering code
X = []
for x in test_set_top_5:
    X.append(x['cuisine'])
print(X)
# print(X)
# del X['ingredients']
# print(X)
Y = []

for y in test_set_top_5:
    Y.append(y['ingredients'])

# Training: Choose your poison
# Create and fit your model on the training data.
model = DecisionTreeClassifier()
model.fit(X, Y)
tree.export_graphviz(model, out_file='ingredients.dot',
                     feature_names=['cuisine', 'cuisine'],
                     class_names=sorted(labels_unique),
                     label='all',
                     rounded=True,
                     filled=True
                     )

# Evaluation:
## Performance Metrics
# write code to compute your preferred performance metrics on training and test data

## Confusion Matrix
# write code to plot a confusion matrix

## ROC Curve
# write code to draw a ROC curve per class in your model
