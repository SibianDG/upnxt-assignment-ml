# Assignment: Cuisine Classifier

Sibian De Gussem

Imagine we're building a new cooking digital product to organize your recipes. In this project, you're building a cuisine classifier to adapt the recipe page UI depending on the cuisine type.

** The goal of the ML assignment is to predict the cuisine of a recipe by its ingredients.**



## Practical Instructions
- You can fork this repository to your account and push your results to a feature branch that you can share back with us. 
- We expect you to use Python 3.8, Jupyter Notebooks, and add your dependencies to requirement.txt in the repository.
- You can submit your assignment as a notebook, we will have a look at your graphs via github, and we'll rerun your code locally.
- The assignment should take around three hours to complete.


## Data

There are two files with data
- train_set_top_5.pkl
- test_set_top_5.pkl

They both have the same following structure.

- id:str recipe identifier
- cuisine:str recipe cuisine
- ingredients:[str] all ingredients of the recipe

Example item:

```
[{'id': 14215,
  'cuisine': 'italian',
  'ingredients': ['bread crumbs',
   'large eggs',
   'all-purpose flour',
   'fresh basil',
   'crushed tomatoes',
   'shallots',
   'dried oregano',
   'mozzarella cheese',
   'grated parmesan cheese',
   'garlic cloves',
   'boneless chicken thighs',
   'olive oil',
   'crushed red pepper flakes']},

   ...

   ```