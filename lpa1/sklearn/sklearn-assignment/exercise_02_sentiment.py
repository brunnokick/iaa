# %%
import random
import pandas as pd
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# %%
random.seed(42)
np.random.seed(42)


# %% [markdown]
## Common Funcions

# %%
def get_train_test_data(verbose=True, 
    data_path=r"lpa1/sklearn/sklearn-assignment/data"):
    movie_reviews_data_folder = data_path
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42)
    if (verbose):
        print(f"n_samples: {len(dataset.data)}")
        print(f"Train data: features: {len(x_train)} | target: {len(y_train)}")
        print(f"Test data: features: {len(x_test)} | target: {len(y_test)}")
    return x_train, x_test, y_train, y_test


# %%
def grid_search(model, x_train, y_train):
    gs = GridSearchCV(model(), model.params(), n_jobs=-1)
    gs = gs.fit(x_train, y_train)
    print(f'Best score: {gs.best_score_} \n Best Params: {gs.best_params_}')
    # results = pd.DataFrame(gs.cv_results_()
    return gs


# %%
def print_metrics(model, predicted, y_test):
    print(f'Acertos: {round(np.mean(predicted == y_test) * 100,2)}%')
    print()
    print("Classification report for classifier %s:\n%s\n"
          % (model, metrics.classification_report(y_test, predicted)))
    print()
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))


# %% [markdown]
## Pipelines
# %%
class ModelBase:
    def __init__(self):
        self.model = None

    def __call__(self):
        return self.model

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)


# %% [markdown]
### Pipeline A
# %%
class Model_A(ModelBase):
    def __init__(self):
        super().__init__()
        self.model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())])

    def params(self):
        return {
            'tfidf__norm': ['l1', 'l2', 'max'],
            'tfidf__use_idf': (False, True),
            'tfidf__smooth_idf': (False, True),
            'tfidf__sublinear_tf': (False, True),
            # 'clf__alpha': [v/10 for v in range(11)], não é possível usar alpha < 1.0
            'clf__fit_prior': (False, True)
        }


# %% [markdown]
### Pipeline B
# %%
class Model_B(ModelBase):
    def __init__(self):
        super().__init__()
        self.model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LinearSVC())])

    def params(self):
        return {
            'tfidf__norm': ['l1', 'l2', 'max'],
            'tfidf__use_idf': (False, True),
            'tfidf__smooth_idf': (False, True),
            'tfidf__sublinear_tf': (False, True),
            'clf__penalty' : ['l1', 'l2'],
            'clf__loss' : ['hinge', 'squared_hinge'],
            'clf__dual': (False, True),
            #'clf__C' : [.0, .5, .1],
            #'clf__multi_class ': ['ovr', 'crammer_singer']
        }

# %% [markdown]
### Pipeline C
# %%
class Model_C(ModelBase):
    def __init__(self):
        super().__init__()
        self.model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', DecisionTreeClassifier())])

    def params(self):
        return {
            'tfidf__norm': ['l1', 'l2', 'max'],
            'tfidf__use_idf': (False, True),
            'tfidf__smooth_idf': (False, True),
            'tfidf__sublinear_tf': (False, True),
            'clf__criterion' : ["gini", "entropy"],
            'clf__splitter' : ["best", "random"]
        }


# %% [markdown]
## Runner

# %%
x_train, x_test, y_train, y_test = get_train_test_data(data_path=r'data/')

# %% [markdown]
### Modelo A

# %%
# Default Params
model_a = Model_A()
model_a.fit(x_train, y_train)
predicted = model_a.predict(x_test)
print_metrics(model_a, predicted, y_test)

# %%
# Best Features
gs = grid_search(model_a, x_train, y_train)
best_params_predicted = gs.predict(x_test)
print_metrics(gs, best_params_predicted, y_test)


# %% [markdown]
### Modelo B 

# %%
# Default Params
model_b = Model_B()
model_b.fit(x_train, y_train)
predicted = model_b.predict(x_test)
print_metrics(model_b, predicted, y_test)

# %%
# Predict with best features
gs = grid_search(model_b, x_train, y_train)
best_params_predicted = gs.predict(x_test)
print_metrics(gs, best_params_predicted, y_test)

# %% [markdown]
### Modelo C

# %%
# Default Params
model_c = Model_C()
model_c.fit(x_train, y_train)
predicted = model_c.predict(x_test)
print_metrics(model_c, predicted, y_test)

# %%
# Predict with best features
gs = grid_search(model_c, x_train, y_train)
best_params_predicted = gs.predict(x_test)
print_metrics(gs, best_params_predicted, y_test)