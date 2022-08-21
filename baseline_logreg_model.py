# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 08:44:37 2022

@author: RUI.WANG2
"""
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE 
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import pandas as pd


#n-gram analysis
'''
data = pd.read_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\dataset.pkl")
bow_converter = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
x = bow_converter.fit_transform(data['NOTES'])
words = bow_converter.get_feature_names_out()


bigram_converter = CountVectorizer(tokenizer=lambda doc: doc, ngram_range=[2,2], lowercase=False) 
x2 = bigram_converter.fit_transform(data['NOTES'])
bigrams = bigram_converter.get_feature_names_out()


trigram_converter = CountVectorizer(tokenizer=lambda doc: doc, ngram_range=[3,3], lowercase=False) 
x3 = trigram_converter.fit_transform(data['NOTES'])
trigrams = trigram_converter.get_feature_names_out()
'''
#to get graph of n-grams
'''
sns.set_style("white")
counts = [len(words), len(bigrams), len(trigrams)]
plt.plot(counts, color='blue')
plt.plot(counts, 'bo')
#plt.margins(0.1)
plt.ticklabel_format(style = 'plain')
plt.xticks(range(3), ['unigram', 'bigram', 'trigram'])
plt.tick_params(labelsize=14)
plt.title('Number of ngrams in tNPS Personal/Business Auto Claims', {'fontsize':16})
plt.show()
'''
#use SMOTE because of class imbalance in dataset -> overwhelming majority of Promoter means model will predict Promoter on every single claim
#SMOTE with re_sample() helps adjust for this imbalance
oversample = SMOTE()
#training_data, test_data = train_test_split(data, train_size = 0.75, random_state = 0)
training_data = pd.read_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\training.pkl")
test_data = pd.read_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\testing.pkl")
#training_data.to_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\training.pkl")
#test_data.to_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\testing.pkl")

#bag-of-words transform on NOTES
bow_transform = CountVectorizer(tokenizer=lambda doc: doc, ngram_range=[3,3], lowercase=False) 
X_tr_bow = bow_transform.fit_transform(training_data['NOTES'])
X_te_bow = bow_transform.transform(test_data['NOTES'])

#TF-IDF transform on NOTES
tfidf_transform = TfidfTransformer(norm=None)
X_tr_tfidf = tfidf_transform.fit_transform(X_tr_bow)
X_te_tfidf = tfidf_transform.transform(X_te_bow)


#set target var to be the designation and not the actual score.
y_tr = training_data['LTR_DESIGNATION']
y_te = test_data['LTR_DESIGNATION']

#adjusting for class imbalance
X_tr_bow, y_tr_bow = oversample.fit_resample(X_tr_bow, y_tr)
X_tr_tfidf, y_tr_tfidf = oversample.fit_resample(X_tr_tfidf, y_tr)

#very simple baseline log reg model
def logistic_classify(X_tr, y_tr, X_test, y_test, description, _C=1.0):
    model = LogisticRegression(C=_C, solver='lbfgs').fit(X_tr, y_tr)
    score = model.score(X_test, y_test)
    print('Test Score with', description, 'features', score)
    return model

model_bow = logistic_classify(X_tr_bow, y_tr_bow, X_te_bow, y_te, 'bow')
model_tfidf = logistic_classify(X_tr_tfidf, y_tr_tfidf, X_te_tfidf, y_te, 'tf-idf')

#grid search to find the best parameters
param_grid_ = {'C': [1e-4, 1e-3, 1e-1, 1e0, 1e1, 1e2]}
bow_search = GridSearchCV(LogisticRegression(), cv=5, param_grid=param_grid_)
tfidf_search = GridSearchCV(LogisticRegression(), cv=5, param_grid=param_grid_)
bow_search.fit(X_tr_bow, y_tr_bow)
#bow_search.best_score_

tfidf_search.fit(X_tr_tfidf, y_tr_tfidf)
#tfidf_search.best_score_

#bow_search.best_params_
#tfidf_search.best_params_

#bow_search.cv_results_


results_file = open('tfidf_gridcv_results.pkl', 'wb')
pickle.dump(bow_search, results_file, -1)
pickle.dump(tfidf_search, results_file, -1)
results_file.close()


pkl_file = open('tfidf_gridcv_results.pkl', 'rb')
bow_search = pickle.load(pkl_file)
tfidf_search = pickle.load(pkl_file)
pkl_file.close()

#table of accuracies
search_results = pd.DataFrame.from_dict({'bow': bow_search.cv_results_['mean_test_score'],
                               'tfidf': tfidf_search.cv_results_['mean_test_score']})

#accuracy box plot.
ax = sns.boxplot(data=search_results, width=0.4)
ax.set_ylabel('Accuracy', size=14)
ax.tick_params(labelsize=14)
plt.savefig('tfidf_gridcv_results.png')

#running the model again on the best parameters
model_bow = logistic_classify(X_tr_bow, y_tr_bow, X_te_bow, y_te, 'bow', _C=bow_search.best_params_['C'])
model_tfidf = logistic_classify(X_tr_tfidf, y_tr_bow, X_te_tfidf, y_te, 'tf-idf', _C=tfidf_search.best_params_['C'])


#generating confusion matrices
y_pred_bow = model_bow.predict(X_te_bow)
cnf_matrix_bow = metrics.confusion_matrix(y_te,  y_pred_bow)

y_pred_tfidf = model_tfidf.predict(X_te_tfidf)
cnf_matrix_tfidf = metrics.confusion_matrix(y_te, y_pred_tfidf)


fig, ax2 = plt.subplots()
class_names=[0,1] # name of classes
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix_bow), annot=True, cmap="YlGnBu" ,fmt='g')
ax2.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix bow', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

sns.heatmap(pd.DataFrame(cnf_matrix_tfidf), annot = True, cmap ='YlGnBu', fmt = 'g')
ax2.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix tfidf', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#performance metrics
print("Accuracy, bow:", metrics.accuracy_score(y_te, y_pred_bow))
print("Precision, bow:", metrics.precision_score(y_te, y_pred_bow))
print("Recall, bow:", metrics.recall_score(y_te, y_pred_bow))
print("AUC, bow:", metrics.roc_auc_score(y_te, y_pred_bow))
print("f-1, tfidf:", metrics.f1_score(y_te, y_pred_bow))

print("Accuracy, tfidf:", metrics.accuracy_score(y_te, y_pred_tfidf))
print("Precision, tfidf:", metrics.precision_score(y_te, y_pred_tfidf))
print("Recall, tfidf:", metrics.recall_score(y_te, y_pred_tfidf))
print("AUC, tfidf:", metrics.roc_auc_score(y_te, y_pred_tfidf))
print("f-1, tfidf:", metrics.f1_score(y_te, y_pred_tfidf))