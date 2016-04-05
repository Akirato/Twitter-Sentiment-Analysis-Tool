import gensim,numpy
from nltk.stem.porter import *
import nltk
import logging
import csv,pickle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SoftmaxLayer
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.Word2Vec.load('model.en')
print "The model is loaded."
stemmer = PorterStemmer()

def vectorize(sentence):
    print sentence
    count = 0 
    a = open('unknown_word').read().strip().lower()
    unsolved_default_vector = numpy.concatenate((model[a],model[a]),axis=0)
    vec_list = []
    try:
        words = [e.lower().encode('ascii','ignore') for e in nltk.word_tokenize(sentence.strip()) if len(set(e))>=3]
    except:
        words = [e.lower() for e in sentence.split() if len(set(e))>=3]
    print words
    if len(words)==1:
        try:
            vec_list.append(numpy.concatenate((model[words[0]],model[words[0]]),axis=0))
        except:
            count+=1
            vec_list.append(unsolved_default_vector)
    elif len(words)==0:
        try:
            vec_list.append(unsolved_default_vector)
        except:
            count+=1
            vec_list.append(unsolved_default_vector)
    else:
        for ind in range(len(words)-1):
            try:
                vec_list.append(numpy.concatenate((model[words[ind]],model[words[ind+1]]),axis=0))
            except:
                count+=1
                vec_list.append(unsolved_default_vector)
    try:
        return sum(vec_list)/len(vec_list)
    except:
        print "Hugeeee Error:",vec_list
        exit()

def test_rnn(train_arrays,train_labels,test_arrays,test_labels):
    ds = SupervisedDataSet(400,1)
    for i in range(len(train_arrays)):
        a,b = train_arrays[i].tolist(),train_labels[i].tolist()
	ds.addSample(a,b)
    for i in range(len(test_arrays)):
        a,b = test_arrays[i].tolist(),test_labels[i].tolist()
        ds.addSample(a,b)
    net = buildNetwork(400, 3, 1, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(net, ds)
    print "Training Error:"
    print trainer.train()
    print "Training Until Convergence"
    print trainer.trainUntilConvergence(maxEpochs=20,verbose=True)
    print trainer.testOnData(verbose=True)
    pickle.dump( net, open( "bigram_rnn_predictor.pickle", "wb" ) )


def test_classifiers(train_arrays,train_labels,test_arrays,test_labels):
    # Logistic Regression
    classifier = LogisticRegression()
    classifier.fit(train_arrays, train_labels)
    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
             intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
    print "Logistic Regression Classifier:",classifier.score(test_arrays, test_labels)
    pickle.dump( classifier, open( "bigram_logistic_predictor.pickle", "wb" ) )
    
    # Naive Bayes
    classifier = GaussianNB()
    classifier.fit(train_arrays, train_labels)
    GaussianNB()
    print "Naive Bayes Classifier:",classifier.score(test_arrays, test_labels)
    pickle.dump( classifier, open( "bigram_nb_predictor.pickle", "wb" ) )

    # Decision Trees
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(train_arrays, train_labels)
    tree.DecisionTreeClassifier()
    print "Decision Tree Classifier:",classifier.score(test_arrays, test_labels)
    pickle.dump( classifier, open( "bigram_dt_predictor.pickle", "wb" ) )

    # Random Forest
    classifier = RandomForestClassifier(n_estimators = 100)
    classifier.fit(train_arrays, train_labels)
    RandomForestClassifier()
    print "Random Forest Classifier:",classifier.score(test_arrays, test_labels)
    pickle.dump( classifier, open( "bigram_ran_for_predictor.pickle", "wb" ) )

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
    pickle.dump( classifier, open( "bigram_neuralnet_predictor.pickle", "wb" ) )

    # Support Vector Machines - Does not support big data ... is of no use.
#    classifier = svm.SVC()
#    classifier.fit(train_arrays, train_labels)
#    svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#        decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#        max_iter=-1, probability=False, random_state=None, shrinking=True,
#        tol=0.001, verbose=False)
#    print "Support Vector Machines Classifier:",classifier.score(test_arrays, test_labels)
#    pickle.dump( classifier, open( "bigram_svm_predictor.pickle", "wb" ) )
    return None

def train(corpus):
    fp = open(corpus, 'rb' )
    reader = csv.reader( fp, delimiter=',', quotechar='"', escapechar='\\' )
    train_len,test_len = 180000,20000
    train_arrays,train_labels = numpy.zeros((train_len,400)),numpy.zeros(train_len)
    test_arrays,test_labels = numpy.zeros((test_len,400)),numpy.zeros(test_len)
    count = 0
    for row in reader:
        m = 1 if int(row[0])>0 else 0
        if count < train_len:
            train_arrays[count] = vectorize(row[5])
            train_labels[count] = m
        else:
            test_arrays[count-train_len] = vectorize(row[5])
            test_labels[count-train_len] = m
        count+=1
        if count>=(train_len+test_len):
	    break
    test_classifiers(train_arrays,train_labels,test_arrays,test_labels)
    print "Done making Arrays"
    print "Starting Testing"
    print "RNN:"
    test_rnn(train_arrays,train_labels,test_arrays,test_labels)
    
if __name__ == '__main__':
    train('Data/sentiment.csv')
