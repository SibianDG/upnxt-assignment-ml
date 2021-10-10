import matplotlib.pyplot as plt

import collections

import pickle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer as vectorizer
from sklearn.linear_model import LinearRegression as regressor

with open('train_set_top_5.pkl', 'rb') as f:
    train_set_top_5 = pickle.load(f)
train_set_top_5 = np.array(train_set_top_5)
print(train_set_top_5[:3])

with open('test_set_top_5.pkl', 'rb') as f:
    test_set_top_5 = pickle.load(f)
test_set_top_5 = np.array(test_set_top_5)

labels = [el["cuisine"] for el in train_set_top_5]
labels_unique = set(labels)
# print("cuisines: ", labels)


number_of_cuisines = sorted(collections.Counter(labels).items(), key=lambda x: x[1], reverse=True)
# print('no_of_ingredients', no_of_ingredients)
# histogram
plt.hist(labels, color='green', histtype='bar')
plt.xlabel('Cuisine')
plt.ylabel('#')
plt.title('Histogram no. of cuisines')
plt.show()

# no. of Ingredients
no_of_ingredients = [len(el['ingredients']) for el in train_set_top_5]

plt.hist(no_of_ingredients, color='green', histtype='bar')
plt.xlabel('ingredients')
plt.ylabel('#')
plt.title('no. of Ingredients')
plt.show()

# avg number of ingredients by cuisine
avg_ingredients = {}
for label in labels_unique:
    avg_ingredients[label] = 0
for el in train_set_top_5:
    avg_ingredients[el['cuisine']] += len(el['ingredients'])
for c in number_of_cuisines:
    avg_ingredients[c[0]] = avg_ingredients[c[0]] / c[1]
print(avg_ingredients)
avg_numbers = [value for key, value in avg_ingredients.items()]
print(avg_numbers)

plt.bar(list(labels_unique), avg_numbers, width=0.8, color=['red', 'green'])
plt.xlabel('cuisine')
plt.ylabel('avg')
plt.title('avg number of ingredients by cuisine')
plt.show()

# remove dished with only 1 ingredient:
train_set_top_5_filtered = [x for x in train_set_top_5 if len(x['ingredients']) > 1]
print(len(train_set_top_5), "--", len(train_set_top_5_filtered))

test_list = ['eggs', 'salt', 'peper', 'chili', 'crab', 'onion', 'oil', 'corn starch']
test_list_final = []
test_list_string = ""

for x, y in enumerate(test_list):
    test_list_string = test_list_string + y + ""
test_list_final.append(test_list_string)

vectors2 = vectorizer.transform(test_list_final)
feature_names = vectorizer.get_feature_names()

vectors_array_test = vectors2.toarray()
df_test = pd.DataFrame(vectors_array_test, columns=feature_names)

test_predict = regressor.predict(df_test)
print(test_predict)