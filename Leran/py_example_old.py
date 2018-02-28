from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from math import floor
from random import sample
import pandas as pd
import sys
import numpy as np
import time
import pickle as pk

# gets list of strings from column colname in dataframe df
def get_text(colname,df):
    stringsList = []
    columnList = list(df[colname])
    print "columnList--------"
    print columnList
    print "range(len(columnList))------"
    print range(len(columnList))
    print "len(columnList)------"
    print len(columnList)
    for i in range(len(columnList)):
        print type(columnList[i])
        if type(columnList[i]) != float:
            text = columnList[i]
            print "text------"
            print text,type(columnList[i])
            stringsList.append(text.lower().replace('\n','').replace('\r',''))
        else:
            stringsList.append('else')
    return stringsList

# appends strings across row from given columns colnames in dataframe df
def get_multitext(df,*colnames):
    zippedStrings = []
    for colname in colnames:
        colstrings = get_text(colname,df)
        print colstrings
        if len(zippedStrings) != 0:
            zippedStrings = zip(colstrings,zippedStrings)
            print "zippedStrings"
            print zippedStrings
            zippedStrings = ['*'.join(x) for x in zippedStrings].collect()
            print zippedStrings
        else:
            zippedStrings = colstrings
    return zippedStrings

# returns percentage of correct predictions from lists of correct and incorrect ratings
def average_correct(correct,incorrect):
    return float(sum(correct))/(sum(correct)+sum(incorrect))

# gets include/exclude class labels from column colname in dataframe df
# verbose boolean allows for longer class names (e.g. Include - Auto)
def get_labels(colname,verbose,df):
    labels = []
    sourceList = list(df[colname])
    if verbose:
        for i in range(len(sourceList)):
            if type(sourceList[i]) != float:
                labels.append(sourceList[i].lower())
            else:
                labels.append(u'')
    else:
        for i in range(len(sourceList)):
            if type(sourceList[i]) != float:
                labels.append(sourceList[i].lower().split(' ',1)[0])
            else:
                labels.append(u'')
    return labels

# splits input set into testing and training sets at given proportion
# trainingProp >0 and <1 => training set is trainingProp percent of inputSet
# trainingProp >1 => training set is the decimal portion of trainingProp percent of inputSet, capped at whole number portion
# trainingProp <=0 => training set = testing set = inputSet
def sample_sets(inputSet, trainingProp):
    numOfElements = len(inputSet)
    if trainingProp < 1.0 and trainingProp > 0.0:
        if numOfElements <= 5:
            trainingSet = inputSet
            testingSet = inputSet
        else:
            trainingLen = int(floor(trainingProp*numOfElements))
            trainingSet = set(sample(inputSet,trainingLen))
            testingSet = inputSet - trainingSet
    elif trainingProp > 1.0:
        cap = int(floor(trainingProp))
        prop = trainingProp % 1
        if numOfElements <= 5:
            trainingSet = inputSet
            testingSet = inputSet
        elif numOfElements*prop > cap:
            trainingSet = set(sample(inputSet,cap))
            testingSet = inputSet - trainingSet
        else:
            trainingLen = int(floor(prop*numOfElements))
            trainingSet = set(sample(inputSet,trainingLen))
            testingSet = inputSet - trainingSet
    else:
        trainingSet = inputSet
        testingSet = inputSet
    return list(trainingSet), list(testingSet)

# gets tuple of strings, labels, and complaint ids for analysis from dataframe df
def get_labeled_tuple_for_prediction(df):
    stringsList = get_multitext(df,'one','three')
    labels = get_labels('two',False,df)
    for i in range(len(labels)):
        if labels[i] == '1.':
            continue
        elif labels[i] == '':
            continue
        else:
            labels[i] = 'exclude'
    ids = df['one']
    tupleList = zip(stringsList,labels,ids)
    labeledTupleList = [x for x in tupleList if (x[0] != '' and x[1] != '')]

    return labeledTupleList

# gets tuple of strings, dummy blank labels, and complaint ids for analysis from dataframe df
def get_unlabeled_tuple_for_prediction(df):
    stringsList = get_multitext(df,'Client Feedback','Resolution Comments', 'Analysis Comments')
    print stringsList
    blankLabels = []
    for i in range(len(stringsList)):
        blankLabels.append('')
    ids = df['Case Number']
    print ids
    tupleList = zip(stringsList,blankLabels,ids)
    print tupleList
    tupleList = [x for x in tupleList if x[0] != '']

    return tupleList

d = {'one' : pd.Series(["ab c","D$EF","xyZ",12.2], index=['a', 'b', 'c','d']), 'two' : pd.Series(["1.", "2.", "3.", "4."], index=['a', 'b', 'c', 'd']),
'three' : pd.Series(["hello","hi how ru","hello world","heyy"],index=['a','b','c','d'])}
df = pd.DataFrame(d)
print df

# gets data from dataframe for prediction
newTuple = get_unlabeled_tuple_for_prediction(df) 
print newTuple
#colString = get_text("one",df)
#print colString

#col1String=get_multitext(df,"one","two","three")
#print col1String

#label=get_labels("one",True,df)
#print label
#label2=get_labels("one",False,df)
#print label2
#get_labeled_tuple_for_prediction(df)
#print labeledTupleList
