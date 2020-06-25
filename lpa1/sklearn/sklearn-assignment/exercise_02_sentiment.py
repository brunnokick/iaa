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

# %%
random.seed(42)
np.random.seed(42)

# %%
def get_train_test_data(verbose=True):
    data_path = r"lpa1/sklearn/sklearn-assignment/data"
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
x_train, x_test, y_train, y_test = get_train_test_data()

# %%
pipeline_a = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

pipeline_a.fit(x_train, y_train)
predicted = pipeline_a.predict(x_test)

print(f'Acertos: {np.mean(predicted == y_test) * 100}%')
print('\n\n')
print("Classification report for classifier %s:\n%s\n"
      % (pipeline_a, metrics.classification_report(y_test, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))
