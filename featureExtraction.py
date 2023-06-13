# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 2022

@author: Euan Goodbrand
"""

#Imports deafult dictionary to avoid errors from dictionary
from collections import defaultdict

#Numpy imported to code the count vectoriser
import numpy as np

class FeatureExtraction:

    def __init__(self, words):

        self.features = words

    #Unused Count Vectoriser
    def count_vectorizer(self, sentences):
        """Count the number of occurrences of each unique word in the given sentences.

        Args:
            sentences (list): List of sentences to count words in.

        Returns:
            dict: A dictionary mapping each unique word to its count.
        """
        # Initialize an empty list to store all words
        all_words = []

        # Iterate over each sentence and word, and append the word to the list
        for sentence in sentences:
            for word in sentence.tokens:
                all_words.append(word)

        # Get the set of unique words
        unique_words = set(all_words)

        # Initialize a dictionary to store word counts
        word_counts = {}

        # Count the number of occurrences of each unique word in the sentences
        for word in unique_words:
            word_counts[word] = np.sum([word in sentence.tokens for sentence in sentences])

        # Return the word counts
        return word_counts

    def sentiment_bias(self, numberSentiments, number_classes):
        """Compute the sentiment bias for sentiments.

        Args:
            numberSentiments (list): List of number of sentiments for each class.
            number_classes (int): Number of classes.

        Returns:
            float: The sentiment bias.
        """
        # Count the number of times each token appears in different sentiment (removed and shortened)
        # token_weights_dict = {
        # "happy": 4,
        # "awe":2,
        # "excellent":4,
        # "joy":4,
        # "great":0.2,
        # "awesome":4,
        # "nice":0.2,
        # "film":0.2,
        # "like":-1.2,
        # "amazing":2}

        # Initialize the bias to 0
        bias = 0

        # Set the weights for each sentiment class
        if number_classes == 3:
            weights = [-1, 0, 1]
        elif number_classes == 5:
            weights = [-2, -1, 0, 1, 2]

        # Iterate over each sentiment class and compute the bias
        for i in range(number_classes):
            bias += numberSentiments[i] * weights[i]
            #Multiple bias by the token weights in the 
            # original_weight = token_weights_dict.get(token, 0)
            # entiment_bias = sentiment_bias * original_weight

        # Return the computed bias
        return bias

    def reduced_sorted_biases(self, biases, featurePercentage):
        """Sort the given biases and return a reduced list.

        Args:
            biases (list): List of biases to sort.
            featurePercentage (float): The percentage of biases to return.

        Returns:
            list: The sorted and reduced list of biases.
        """
        # Sort the biases in descending order
        sortedBiases = sorted(biases, key=lambda item: item[1] * item[1], reverse=True)

        # Compute the number of biases to return
        amountOfFeatures = int(len(biases) * featurePercentage)

        # Return the sorted and reduced list of biases
        return sortedBiases[:amountOfFeatures]

    def featureExtraction(self, words, number_classes):
        """Extract features.

        Args:
            words (list): List of words to extract features from.
            number_classes (int): Number of classes to extract features for.

        Returns:
            list: List of words with filtered tokens.
        """
        # Initialize empty list to store sentiment biases
        biases = []

        # Set feature percentage
        featurePercentage = 0.38
        if number_classes == 3:
            featurePercentage = 0.97

        # Count occurrences of tokens in each sentiment
        tokenSentiments = defaultdict(lambda: [0] * number_classes)
        for word in words:
            sentiment = word.sentiment
            for token in word.tokens:
                tokenSentiments[token][sentiment] = 1 + tokenSentiments[token][sentiment]

        # Compute sentiment biases for each token
        for token, numberSentiments in tokenSentiments.items():
            bias = self.sentiment_bias(numberSentiments, number_classes)
            biases.append((token, bias))

        # Reduce and sort sentiment biases
        reducedBiases = self.reduced_sorted_biases(biases, featurePercentage)

        # Set the features to be the top tokens by bias
        self.features = [token for token, _ in reducedBiases]

        # Filter words by keeping only tokens that are in the features
        filtered_words = []
        for word in words:
            filtered_tokens = [token for token in word.tokens if token in self.features]
            if filtered_tokens:
                word.tokens = filtered_tokens
                filtered_words.append(word)

        #Count vectorising word list removed as didn't improve evaluation score.
        # word_counts = self.count_vectorizer(words)
        # words_to_keep = []
        # for word, count in word_counts.items():
        #     if count > 4 and count < 10:
        #         words_to_keep.append(word)
        
        # self.features = words_to_keep
        # return words_to_keep

        # Return the filtered words
        return filtered_words
