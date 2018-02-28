# import statements
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from math import floor, ceil
from random import sample, uniform
import pandas as pd
import sys
import numpy as np
import time
import pickle as pk

# gets list of strings from column colname in dataframe df
def get_text(colname,df):
    stringsList = []
    columnList = list(df[colname])
    for i in range(len(columnList)):
        if type(columnList[i]) != float:
            text = columnList[i]
            stringsList.append(text.lower().replace('\n','').replace('\r',''))    
        else:
            stringsList.append(u'')
    return stringsList

# appends strings across row from given columns colnames in dataframe df
def get_multitext(df,*colnames):
    zippedStrings = []
    for colname in colnames:
        colstrings = get_text(colname,df)
        if len(zippedStrings) != 0:
            zippedStrings = zip(colstrings,zippedStrings)
            zippedStrings = [' '.join(x) for x in zippedStrings]
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
    stringsList = get_multitext(df,'Client Feedback','Resolution Comments', 'Analysis Comments')
    labels = get_labels('Bucket',False,df)
    for i in range(len(labels)):
        if labels[i] == 'ana':
            continue
        elif labels[i] == '':
            continue
        else:
            labels[i] = 'exclude'
    ids = df['Case Number']
    tupleList = zip(stringsList,labels,ids)
    labeledTupleList = [x for x in tupleList if (x[0] != '' and x[1] != '')]
    
    return labeledTupleList

# gets tuple of strings, dummy blank labels, and complaint ids for analysis from dataframe df
def get_unlabeled_tuple_for_prediction(df):
    stringsList = get_multitext(df,'Client Feedback','Resolution Comments', 'Analysis Comments')
    blankLabels = []
    for i in range(len(stringsList)):
        blankLabels.append('')
    ids = df['Case Number']
    tupleList = zip(stringsList,blankLabels,ids)
    tupleList = [x for x in tupleList if x[0] != '']
    
    return tupleList

# prepares data from the sampling stage to the tfidf fitting stage using dataframe df
# ngramMax is the largest ngram size (should use 3)
# min_dof and max_dof are the bounds which an ngram must fit in to be included
# e.g. max_dof = .4 excludes ngrams that exist in more than 40% of the population
def data_prep(trainProp,ngramMax,min_dof,max_dof,df):
    # gets list of tuples for analysis
    labeledTupleList = get_labeled_tuple_for_prediction(df)

    # gets sets for cross-validation
    trainList,testList = sample_sets(set(labeledTupleList),trainProp)
    
    # breaks tuples apart
    trainText = [x[0] for x in trainList]
    testText = [x[0] for x in testList]
    trainLabels = [x[1] for x in trainList]
    testLabels = [x[1] for x in testList]
    trainIds = [x[2] for x in trainList]
    testIds = [x[2] for x in testList]

    # creates and fits countvectorizer based on training set
    CV = CountVectorizer(ngram_range=(1,ngramMax),min_df=min_dof,max_df=max_dof)
    trainCounts = CV.fit_transform(trainText)

    # creates and fits tfidf based on training set and countvectorizer
    tfidfTransformer = TfidfTransformer().fit(trainCounts)
    fitTfidf = tfidfTransformer.transform(trainCounts)
    
    return CV,tfidfTransformer,fitTfidf,trainList,testList

# general SVM classifier object, still using linear kernel
# tied for lead in accuracy and allows for confidence (0 through 1) output
def general_svm_classifier(CV,tfidfTransformer,fitTfidf,model,testList):
    # breaks tuples apart
    testText = [x[0] for x in testList]
    testLabels = [x[1] for x in testList]
    testIds = [x[2] for x in testList]
    
    correct = 0
    incorrect = 0
    incCount = 0
    exCount = 0
    predList = []

    # predicts using classifier
    for i in range(len(testText)):
        text = testText[i]
        correctLabel = testLabels[i]
        CVTransform = CV.transform([text])
        tfidfLoop = tfidfTransformer.transform(CVTransform)
        pred = generalSVCClassifier.predict_proba(tfidfLoop)[0][0]

        predList.append(pred)
        if pred > 0.5:
            pred = 'include'
        else:
            pred = 'exclude'
        
        if pred == 'include':
            incCount += 1
        elif pred == 'exclude':
            exCount += 1
        if pred == correctLabel:
            correct += 1
        else:
            incorrect += 1

    print('include: ',incCount)
    print('exclude: ',exCount)
    print('')
    return predList

# determines whether or not to review a complaint
def review_complaint(dfIn,eightTop,eightBottom,fiftysevenTop,fiftysevenBottom,sampleRate):
    columns = list(dfIn.columns.values)
    if 'CATEGORIES_8_57' in columns:
        review = []
        dfSub = dfIn[['Prediction','CATEGORIES_8_57']]
        dfSub.insert(0,'Case Number',dfIn.index.tolist())
        subTuple = [tuple(x) for x in dfSub.values]
        
        eightTuple = [x for x in subTuple if x[2] == 8]
        eightYesNo = [x for x in eightTuple if (x[1] >= eightTop or x[1] < eightBottom)]
        eightYesNo = set(eightYesNo)
        lenEight = len(eightYesNo)
        eightRandomTrue = sample(eightYesNo,ceil(sampleRate*lenEight))
        eightRandomFalse = eightYesNo - set(eightRandomTrue)
        eightRandomTrue = [(list(x)+[True]) for x in list(eightRandomTrue)]
        eightRandomFalse = [(list(x)+[False]) for x in list(eightRandomFalse)]
        eightGrey = set(eightTuple) - eightYesNo
        eightGrey = [(list(x)+[True]) for x in list(eightGrey)]
        eightFull = eightRandomTrue + eightRandomFalse + eightGrey
        
        fiftysevenTuple = [x for x in subTuple if x[2] == 57]
        fiftysevenYesNo = [x for x in fiftysevenTuple if (x[1] >= fiftysevenTop or x[1] < fiftysevenBottom)]
        fiftysevenYesNo = set(fiftysevenYesNo)
        lenFiftyseven = len(fiftysevenYesNo)
        fiftysevenRandomTrue = sample(fiftysevenYesNo,ceil(sampleRate*lenFiftyseven))
        fiftysevenRandomFalse = fiftysevenYesNo - set(fiftysevenRandomTrue)
        fiftysevenRandomTrue = [(list(x)+[True]) for x in list(fiftysevenRandomTrue)]
        fiftysevenRandomFalse = [(list(x)+[False]) for x in list(fiftysevenRandomFalse)]
        fiftysevenGrey = set(fiftysevenTuple) - fiftysevenYesNo
        fiftysevenGrey = [(list(x)+[True]) for x in list(fiftysevenGrey)]
        fiftysevenFull = fiftysevenRandomTrue + fiftysevenRandomFalse + fiftysevenGrey
        
        incorrectLabelList = [list(x).append('categories label is not in the correct format') for x in subTuple if (x[2] != 8 and x[2]!= 57)]
        
        unorderedList = eightFull + fiftysevenFull + incorrectLabelList
        for i in range(len(subTuple)):
            for j in unorderedList:
                if subTuple[i][0] == j[0]:
                    review.append(j[3])
    else:
        review = ['no categories label'] * len(df['Case Number'])
    return review

# imports data
# df is a dataframe containing the unlabeled testing/evaluation data
# inModel is pickle file containing 4 objects from the ANAModelGenerator.py program
# the objects are CV (a CountVectorizer object), tfidfTransformer, fitTfidf, and generalSVCClassifier (the classifier itself)
inModelName = sys.argv[-3]
inSpreadsheetName = sys.argv[-2]
outName = sys.argv[-1]

df = pd.read_excel(inSpreadsheetName,converters={'CIN':str,'Acct Number':str,'FULL_ACCT_NBR':str,'FEED_CST_CTR_NBR':str,'FEED_DEPT_ID':str,'FEED_EMP_ID':str,'FEED_WRK_ID':str,'Update Account NBR outlined in complaint - use 9999 for CC App D':str,'Update Emp ID as needed':str})

inModel = open(inModelName,'rb')
CV = pk.load(inModel)
tfidfTransformer = pk.load(inModel)
fitTfidf = pk.load(inModel)
generalSVCClassifier = pk.load(inModel)
inModel.close()

# sets threshold parameters
eightTop = 0.8
eightBottom = 0.3
fiftysevenTop = 1
fiftysevenBottom = 0.4
sampleRate = 0.1

# gets data from dataframe for prediction
newTuple = get_unlabeled_tuple_for_prediction(df)

# predicts new data from model imported from pickle file
predictionOutList = general_svm_classifier(CV,tfidfTransformer,fitTfidf,generalSVCClassifier,newTuple)

# adds the output column to the dataframe df
df.set_index('Case Number', inplace=True)
df.insert(13,'Prediction',np.nan)

# adds prediction data to dataframe df
caseNumbers = [x[2] for x in newTuple]
for i in range(len(caseNumbers)):
    df.set_value(str(caseNumbers[i]), 'Prediction', predictionOutList[i])
# inserts review label
df.insert(14,'Review',review_complaint(df,eightTop,eightBottom,fiftysevenTop,fiftysevenBottom,sampleRate))

# writes predictions for data to excel spreadsheet
if outName[-5:] != '.xlsx':
    outName = outName + '.xlsx'
df.to_excel(outName)
