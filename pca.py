# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:16:04 2022

@author: RUI.WANG2
"""

from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 
import pandas as pd

oversample = SMOTE()
#training_data, test_data = train_test_split(data, train_size = 0.75, random_state = 0)
training_data = pd.read_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\training.pkl")
test_data = pd.read_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\testing.pkl")
#training_data.to_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\training.pkl")
#test_data.to_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\testing.pkl")


bow_transform = CountVectorizer(tokenizer=lambda doc: doc, ngram_range=[3,3], lowercase=False) 
X_tr_bow = bow_transform.fit_transform(training_data['NOTES'])
X_te_bow = bow_transform.transform(test_data['NOTES'])


tfidf_transform = TfidfTransformer(norm=None)
X_tr_tfidf = tfidf_transform.fit_transform(X_tr_bow)
X_te_tfidf = tfidf_transform.transform(X_te_bow)

y_tr = training_data['LTR_DESIGNATION']
y_te = test_data['LTR_DESIGNATION']


X_tr_bow, y_tr_bow = oversample.fit_resample(X_tr_bow, y_tr)
X_tr_tfidf, y_tr_tfidf = oversample.fit_resample(X_tr_tfidf, y_tr)

# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
pca = TruncatedSVD() #TruncatedSVD over PCA due to sparse matrix
# Define a Standard Scaler to normalize inputs
scaler = StandardScaler(with_mean= False)

# set the tolerance to a large value to make the example faster
logistic = LogisticRegression(tol=0.1)
pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic", logistic)])



# Parameters of pipelines can be set using '__' separated parameter names:
param_grid = {
    "pca__n_components": [5, 15, 30, 45, 60],
    "logistic__C": np.logspace(-4, 4, 4),
}
search = GridSearchCV(pipe, param_grid, n_jobs=2)
search.fit(X_tr_tfidf, y_tr_tfidf)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

# Plot the PCA spectrum
pca.fit(X_tr_tfidf)

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax0.plot(
    np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, "+", linewidth=2
)
ax0.set_ylabel("PCA explained variance ratio")

ax0.axvline(
    search.best_estimator_.named_steps["pca"].n_components,
    linestyle=":",
    label="n_components chosen",
)
ax0.legend(prop=dict(size=12))

# For each number of components, find the best classifier results
results = pd.DataFrame(search.cv_results_)
components_col = "param_pca__n_components"
best_clfs = results.groupby(components_col).apply(
    lambda g: g.nlargest(1, "mean_test_score")
)

#pca graph
best_clfs.plot(
    x=components_col, y="mean_test_score", yerr="std_test_score", legend=False, ax=ax1
)
ax1.set_ylabel("Classification accuracy (val)")
ax1.set_xlabel("n_components")

plt.xlim(-1, 70)

plt.tight_layout()
plt.show()