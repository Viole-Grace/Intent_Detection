import re
import pandas as pd
import numpy as np
import regex as regex
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from nltk import ngrams
from stemming.porter2 import stem

#import dataset
raw_training_data, raw_testing_data = [],[]

with open("train.txt") as f:
    train=f.readlines()
    ttrain=f.read()
for item in train:
    raw_training_data.append(item.strip())
    
with open("test.txt") as f:
    test=f.readlines()
for item in test:
    raw_testing_data.append(item.strip())

#get an idea of the data
freq=pd.Series(' '.join(train[1]).split()).value_counts()[:10]
freq=list(freq.index)
wordcount = {}
# eliminate duplicates, split by punctuation and use case demiliters
for word in ttrain.lower().split('\t', 1):
    word = word.replace(".","")
    word = word.replace(",","")
    word = word.replace(":","")
    word = word.replace("\"","")
    word = word.replace("!","")
    word = word.replace("*","")
    if word not in wordcount:
        wordcount[word] = 1
    else:
        wordcount[word] += 1
#most common words / features
word_counter = collections.Counter(wordcount)
word_counter
#clean data

def clean(raw_data):
    
    #split into labels and text
    labels=[lab.split('\t', 1)[0] for lab in raw_data]
    training_data= [item.split('\t', 1)[1] for item in raw_data]
    
    labels,training_data
    
    #convert to lowercase, stem / lemmatize
    training_data = [i.lower() for i in training_data]
    training_data = [" ".join([stem(word) for word in sentence.split(" ")]) for sentence in training_data]
        
    #replace links, email_id, currencies, entities etc
    training_data=[re.sub(r'[\w\.-]+@[\w\.-]+', "$EMAIL_ID", i) for i in training_data]
    training_data=[re.sub(r"(<?)http:\S+", "$URL", i) for i in training_data]
    training_data=[re.sub(r"\$\d+", "$CURR", i) for i in training_data]
    training_data=[re.sub(r'\b\d+\b', "$NUM", i) for i in training_data]
    training_data=[re.sub(r'\b(me|her|him|us|them|you)\b', "$ENTITIES", i) for i in training_data]
    
    
    #remove punctuation, special chars, tokenize data
    training_data = [regex.sub(r"[^\P{P}$]+", " ", i) for i in training_data]
    training_data = [re.sub(r"[^0-9A-Za-z/$' ]", " ", i) for i in training_data]
    
    #regularize data w.r.t days, times, months and year
    regex_match_days= r'monday|tuesday|wednesday|thursday|friday|saturday|sunday'
    regex_match_times= r'morning|afternoon|evening'
    regex_match_events= r'after|before|during'
    regex_match_month= r'january|february|march|april|may|june|july|august|september|october|november|december'
    
    training_data = [re.sub(regex_match_days, "$DAY", i) for i in training_data]
    training_data = [re.sub(regex_match_times, "$TIMES", i) for i in training_data]
    training_data = [re.sub(regex_match_events, "$EVENTS", i) for i in training_data]
    training_data = [re.sub(regex_match_month, "$MONTH", i) for i in training_data]
    
    #remove extra spaces and blanks
    training_data = [item.strip() for item in training_data]
    
    #return cleaned data
    return training_data, labels 
#n-gram based SVM Classification, since problem statement says its a binary Classification 
def get_phrases(text, n):
    
    #can define in two ways - one way is to get both bigrams and tri-grams from a single pass of the function
    #depends on the number of values unpacked by the host OS, so it has been rewritten for a dual pass and as a multi n-gram formation function
    
    """#define for 3 n-gram models : unigram, bigram and trigram.
    
    #bi_grams,tri_grams = "",""
    #bi_l,tri_l= [], []
    
    #get ngrams
    
    #bi_grams = ngrams(text.split(), 2)
    #tri_grams = ngrams(text.split(), 3)
    
    #build, set ngrams"""
    """for grams in bi_grams:
        bi_l.append('_'.join(map(str,grams)))
    for grams in tri_grams:
        tri_l.append('_'.join(map(str,grams)))    
    bstring= ' '.join(bi_l)
    tstring= ' '.join(tri_l)    
    return bstring,tstring"""
    
    n_grams=ngrams(text.split(),n)
    gram_list = []
    for grams in n_grams:
        gram_list.append('_'.join(map(str,grams)))
    gram_string = ' '.join(gram_list)
    #print gram_string
    return gram_string
#the research paper attached makes use of TF-IDF, hence ranking constraints will be applied using TF-IDF
def tf_idf(data):
    
    vector_model= TfidfVectorizer(min_df=1) #single dataframe
    X = vector_model.fit_transform(data) #fit model and transform data to required vector
    
    #only one axis holds the ranking -> X axis
    return X
#get the cleaned data : 
training_data, training_labels = clean(raw_training_data)
testing_data, testing_labels = clean(raw_testing_data)

# since labels arent distributed in a random fashion, distribution is either a single set of 'NO' or 'NO' and 'YES' combined.
# get X,Y training data by mixing both, then using k-folds for distribution or check weight biases using jack-knife resampling
# another approach is to mix both data, then split using the train-test-split module.

X = tf_idf(training_data+testing_data)
Y = training_labels+testing_labels

#set random_state to 42 for reproducible results
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#tune parameters for c and gamma, use the rbf kernel
#running the function takes time, so it has already been run once and best C and gammas have been identified
#best C for SVM model - 2100, 14,600, 23,200
def get_c_and_gamma(X,Y,nfolds):
    Cs = [0.0001, 0.001, 0.01, 0.1, 1, 10] #lower Cs value gives a simpler decision function
    gammas = [ 0.001, 0.01, 0.1, 1, 10] #inverse of radius of influence

    from sklearn.grid_search import GridSearchCV
    param_grid= {'C':Cs, 'gamma':gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, Y)
    print grid_search.best_params_
    return grid_search.best_params_
#basic SVM model vs bi-gram model vs tri-gram model
#result classification can be seen using a confusion_matrix

#new_C=get_c_and_gamma(X,Y,10)
svm=SVC(C=2100, kernel = 'rbf')
svm.fit(x_train, y_train)
print "SVM classification : \n"
print "Training Accuracy : ",svm.score(x_train, y_train),"\nTesting Accuracy : ",svm.score(x_test, y_test)
# Training Accuracy : 0.885 , Testing Accuracy : 0.781
#bi-gram and tri-gram SVM model
total_data = training_data+testing_data
total_labels = training_labels+testing_labels

bi_gram_data = [get_phrases(item,2) for item in total_data]
tri_gram_data = [get_phrases(item,3) for item in total_data]

X2, Y2 = tf_idf(bi_gram_data), total_labels
X3, Y3 = tf_idf(tri_gram_data), total_labels

x_train1, x_test1, y_train1, y_test1 = train_test_split(X2, Y2, test_size=0.3, random_state=42)
x_train2, x_test2, y_train2, y_test2 = train_test_split(X3, Y3, test_size=0.3, random_state=42)   

#SVM C value for bi-grams, takes time to compute. Comment it out till the print statement if you dont want to run time constraints
best_c={}
for increment in range(1000, 15000, 100):
    svm=SVC(C=increment, kernel = 'rbf')
    svm.fit(x_train1, y_train1)
    tr_ac, te_ac = svm.score(x_train1, y_train1), svm.score(x_test1, y_test1)
    best_c[str(increment)]=te_ac
import operator
print " Max C value : ",max(best_c.iteritems(),key=operator.itemgetter(1))[0] # 14600

#SVM C value for tri-grams, takes time to compute. Comment it out till the print statement if you dont want to run time constraints
best_tri_c={}
for increment in range(10000, 30000, 100):
    svm=SVC(C=increment, kernel = 'rbf')
    svm.fit(x_train1, y_train1)
    tr_ac, te_ac = svm.score(x_train1, y_train1), svm.score(x_test1, y_test1)
    best_tri_c[str(increment)]=te_ac
import operator
print " Max C value trigrams : ",max(best_tri_c.iteritems(),key=operator.itemgetter(1))[0] # 23200

#bi-gram SVM max value as C = 14600. Calculated using max ( c ) iterative function
svm=SVC(C=14600, kernel='rbf')
svm.fit(x_train1, y_train1)
print "SVM Classification for Bi-Grams : "
print "Training accuracy : ",svm.score(x_train1, y_train1) # 0.9797
print "Testing accuracy : ",svm.score(x_test1, y_test1) # 0.7828

#tri-gram SVM max value as C = 23200. Calculated using max ( c ) iterative function
svm=SVC(C=23200, kernel='rbf')
svm.fit(x_train2, y_train2)
print "\nSVM Classification for Tri-Grams : "
print "Training accuracy : ",svm.score(x_train2, y_train2) # 0.9929
print "Testing accuracy : ",svm.score(x_test2, y_test2) # 0.7391