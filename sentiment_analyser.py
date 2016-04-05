# -*- coding: utf-8 -*-
# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy
import logging
# random
from random import shuffle

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = Doc2Vec.load('./tel.d2v')

train_arrays = numpy.zeros((900, 100))
train_labels = numpy.zeros(900)

for i in range(747):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_labels[i] = 1

for i in range(153):
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[747 + i] = model.docvecs[prefix_train_neg]
    train_labels[747 + i] = 0

#print train_arrays
#print train_labels
test_arrays = numpy.zeros((387, 100))
test_labels = numpy.zeros(387)

for i in range(321):
    prefix_test_pos = 'TEST_POS_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_labels[i] = 1

for i in range(66):
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[321 + i] = model.docvecs[prefix_test_neg]
    test_labels[321 + i] = 0

# Logistic Regression
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
print "Logistic Regression Classifier:",classifier.score(test_arrays, test_labels)

# Support Vector Machines
classifier = svm.SVC()
classifier.fit(train_arrays, train_labels)
svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
print "Support Vector Machines Classifier:",classifier.score(test_arrays, test_labels)

# Naive Bayes
classifier = GaussianNB()
classifier.fit(train_arrays, train_labels)
GaussianNB()
print "Naive Bayes Classifier:",classifier.score(test_arrays, test_labels)

# MLP Neural Network
classifier = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
classifier.fit(train_arrays, train_labels)
MLPClassifier(activation='relu', algorithm='l-bfgs', alpha=1e-05,
       batch_size=200, beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
print "MLP Neural Network Classifier:",classifier.score(test_arrays, test_labels)

# Decision Trees
classifier = tree.DecisionTreeClassifier()
classifier.fit(train_arrays, train_labels)
tree.DecisionTreeClassifier()
print "Decision Tree Classifier:",classifier.score(test_arrays, test_labels)

# Random Forest
classifier = RandomForestClassifier(n_estimators = 100)
classifier.fit(train_arrays, train_labels)
RandomForestClassifier()
print "Random Forest Classifier:",classifier.score(test_arrays, test_labels)

