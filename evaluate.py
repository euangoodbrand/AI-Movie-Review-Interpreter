# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 2022

@author: Euan Goodbrand
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Evaluation:
    """
    This class is used to evaluate the performance of a sentiment analysis model.
    """

    def __init__(self, words, predictions, number_classes):
        """
        Initialize the evaluation with the input words, their predicted sentiments, and the number of sentiment classes in the dataset.

        Args:
            words (List[str]): A list of words.
            predictions (Dict[int, int]): A dictionary that maps a word's sentence ID to its predicted sentiment.
            number_classes (int): The number of sentiment classes in the dataset.
        """
        self.words = words
        self.predictions = predictions
        self.number_classes = number_classes
        
        self.confusion_matrix = self.compute_confusion_matrix()

    def compute_confusion_matrix(self):
        """
        Compute the confusion matrix for the predicted sentiments.

        Returns:
            List[List[int]]: A 2D list representing the confusion matrix.
        """
        words = self.words
        predictions = self.predictions
        number_classes = self.number_classes

        # Initialize the confusion matrix with zeros.
        confusion_matrix = [[0 for j in range(number_classes)] for i in range(number_classes)]

        # Count the number of predictions for each sentiment.
        for word in words:
            predicted_sentiment = predictions.get(word.sentenceID)
            confusion_matrix[word.sentiment][predicted_sentiment] += 1

        return confusion_matrix

    def macro_f1_score(self):
        """
        Calculate the macro-F1 score for the predicted sentiments.

        Returns:
            float: The macro-F1 score.
        """
        # Calculate the precision, recall, and F1 score for each class
        confusion_matrix=self.confusion_matrix
        confusion_matrix = np.array(confusion_matrix)
        f1_scores = []
        for i in range(confusion_matrix.shape[0]):
            true_positives = confusion_matrix[i, i]
            false_positives = confusion_matrix[i].sum() - true_positives
            false_negatives = confusion_matrix[:, i].sum() - true_positives
            
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1_score = 2 * precision * recall / (precision + recall)
            
            f1_scores.append(f1_score)
        
        # Calculate the macro-F1 score
        macro_f1_score = sum(f1_scores) / len(f1_scores)
        return macro_f1_score

    def plot_confusion_matrix(self):
        """ Plot the confusion matrix for the predicted sentiments.  """
        confusion_matrix = self.confusion_matrix

        # Use a seaborn heatmap to visualize the confusion matrix
        df = pd.DataFrame(confusion_matrix)
        ax = sns.heatmap(df, annot=True, fmt="d")
        ax.set_title('Confusion matrix')

        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.show()