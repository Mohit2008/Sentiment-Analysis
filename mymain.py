import pandas as pd
all_data=pd.read_csv('data.tsv', sep=' ', quotechar='"', escapechar='\\',index_col=["new_id"]) # read in the entire data
splits = pd.read_csv("splits.csv", sep='\t') # read the split data
s=1     # set the split fold could be either 1,2,3


import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
warnings.filterwarnings("ignore")
np.random.seed(3064) # set the random seed

import time
start = time.time()
############## Import ends


stops={"i", "me", "my", "myself","we", "our", "ours", "ourselves","you", "your", "yours","their", "they", "his", "her",
               "she", "he", "a", "an", "and","is", "was", "are", "were","him", "himself", "has", "have",
               "it", "its", "of", "one", "for","the", "us", "this"} # set the list of stop words


def clean_text(raw_review):
        review_text = BeautifulSoup(raw_review, features="html5lib").get_text() # remove html tags
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)     # keep only alphabets
        meaningful_words = [w for w in letters_only.lower().split() if w not in stops] # split the words , turn to lowercase and remove stop words
        return(" ".join( meaningful_words )) # return back the cleaned text as a sentence


# function to load in the vocablary for text_2_vec
def get_vocab():
    with open('myVocab.txt', 'r') as f:
        vocab = f.readlines()
    myVocab = [line[:-1] for line in vocab] # remove /n
    return myVocab # list of words


# convery each sentence to vector with the given vocab
def text_2_vec(review_train, review_test, myVocab):
    count_vect = CountVectorizer(ngram_range=(1,2),min_df=0.001,max_df=0.55,vocabulary=myVocab) # set the count vector with params
    X_train_counts = count_vect.fit_transform(review_train.clean_review.tolist()) # fit and transfrom train data
    X_test_counts = count_vect.transform(review_test.clean_review.tolist()) # transform test data
    vectorizer = TfidfTransformer() # us tf-idf
    train_data_features = vectorizer.fit_transform(X_train_counts) # fir and transform train counts
    test_data_features = vectorizer.transform(X_test_counts) # transform test counts
    del X_train_counts, X_test_counts # recover memory
    return train_data_features, test_data_features # return vects


# function to perform modelling
def run_model(train_data_features, review_train,test_data_features,review_test):
    clf = LogisticRegression(C=4,solver ='liblinear', penalty='l2').fit(train_data_features,
                             review_train.sentiment) # do logistic regression with params
    logi_pred=clf.predict_proba(test_data_features) # get the prediction
    pred_test_original= pd.DataFrame(np.round(logi_pred[:,1],4), index=review_test.index) # load prediction in data frame
    pred_test_original.columns=['prob']# set the column name
    pred_test_original.to_csv(out_file_path, header=True, index=True, sep=',')# save to disk
    print("Prediction from model  persisted to disk as mysubmission.txt")


try:
    out_file_path ="mysubmission.txt"
    
    review_test= all_data.loc[splits.iloc[:,s-1], :] # filter test id
    review_train=all_data.drop(review_test.index.values) ## get train data
    
    review_train['clean_review']=review_train['review'].apply(clean_text) # clean all reviews in train
    review_test['clean_review']=review_test['review'].apply(clean_text) # clean all reviews in test
    myVocab=get_vocab() # get my custom vocab
    train_data_features, test_data_features= text_2_vec(review_train, review_test, myVocab) # project to vect
    
    run_model(train_data_features, review_train,test_data_features,review_test) # run the ML model
    print('Script executed succesfully in ', (time.time()-start)/60, ' minutes')

except Exception as ex:
    print("Script halted due to exception in {}".format(ex))