# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
#Import for command line functions
import argparse

#Class imports
#Import functions for preprocessing
from preprocessSentences import *
#Import functions for feature selection
from featureExtraction import *
#Import functions for naiveBayes classification
from naiveBayes import *
#Import functions for evaluation of predictions
from evaluate import *


USER_ID = "arb20eg" 


def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args

 
def main():

    #Command line inputs
    inputs=parse_args()
    
    #Input files 
    training = inputs.training
    dev = inputs.dev 
    test = inputs.test 

    #Number of classes for the sentiment
    number_classes = inputs.classes

    #Choice to use all tokens as features or feature selection
    features = inputs.features
    
    #Option to save predictions
    output_files = inputs.output_files
     
    #Option to display the confusion matrix or not
    confusion_matrix = inputs.confusion_matrix
    
    #Load in initial train data
    words = Sentence.wordList(training)

    #Preprocess the train data

    if number_classes == 3:
        preprocess = Preprocess(words)
        preprocess.preprocessWords()
    
    #Rescale data
    sentiment_scale = [0,1,2,3,4]
    if number_classes == 3:
        sentiment_scale = [0,1,2]
        preprocess.scaleDown()

    #If feature selection is selected then extract features
    if  features == "features":
        featureProcessor = FeatureExtraction(number_classes)
        featureProcessor.featureExtraction(words, number_classes)
        
    #Train the corpora
    corpora = Train(words, sentiment_scale)

    #Load dev and test data
    dev_words = Sentence.wordList(dev)
    test_words = Sentence.wordList(test)

    #Preprocess dev and test data
    if number_classes == 3:
        preprocess_dev = Preprocess(dev_words)
        preprocess_dev.preprocessWords()
    if number_classes == 3:
        preprocess_test = Preprocess(test_words)
        preprocess.preprocessWords()

    #Rescale data
    sentiment_scale = [0,1,2,3,4]
    if number_classes == 3:
        sentiment_scale = [0,1,2]
        preprocess_dev.scaleDown()
        preprocess_test.scaleDown()
        
    #Feature selection for dev and test data
    if  features == "features":
        #Dev data
        featureProcessor = FeatureExtraction(number_classes)
        featureProcessor.featureExtraction(dev_words, number_classes)
        #Test data
        featureProcessor = FeatureExtraction(number_classes)
        featureProcessor.featureExtraction(test_words, number_classes)

    #Predict the sentiments of the dev data
    predict = SentimentPredictor(corpora)
    #Dev predictions
    predicted_dev_results = predict.sentiments(dev_words)
    #Test predictions
    predicted_test_results = predict.sentiments(test_words)

    #File for saving and loading predicit
    dev_file = "dev_predictions_"+str(number_classes)+"classes_"+USER_ID + ".tsv"
    test_file = "test_predictions_"+str(number_classes)+"classes_"+USER_ID + ".tsv"

    #Function to save a temp file for calculating macro when not choosing
    #to save data without output_files command line argument
    dev_no_save_file = "temp_dev_pred.tsv"

    #Save the prediction results if wanted
    if output_files:
        #Save dev predictions
        save_prediction(dev_file, predicted_dev_results, dev_words)
        #Save test predictions
        save_prediction(test_file, predicted_test_results, test_words)

    #Save dev predicitions in temporary file
    save_prediction(dev_no_save_file, predicted_dev_results, dev_words)

    #Load predictions for evaluation
    predictions = load_predictions(dev_no_save_file)

    #Evaluate the prediction results
    evaluation = Evaluation(dev_words, predictions, number_classes)
    f1_score = evaluation.macro_f1_score()

    #Visualise the confusion matrix
    if confusion_matrix:
        evaluation.plot_confusion_matrix()
    
    #Print the results and relevant console information
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()