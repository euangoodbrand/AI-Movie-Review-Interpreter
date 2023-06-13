# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 2022

@author: Euan Goodbrand
"""

# Import the pandas library as pd.
import pandas as pd

def load_predictions(filename):
    """Load predictions from a file.

    Args:
        filename (str): Name of the file to load predictions from.

    Returns:
        dict: A dictionary containing the predictions.
    """
    # Read the file as a CSV and extract the 'Sentiment' column.
    df = pd.read_csv("predictions/"+filename, index_col=0, delimiter='\t')
    return df['Sentiment'].to_dict()

def save_prediction(filename, predictions, new_words):
    """Save predictions to a file.

    Args:
        filename (str): Name of the file to save predictions to.
        predictions (list): List of predicted sentiments.
        new_words (list): List of words that with predictions
    """
    with open("predictions/"+filename, 'w') as f:
        # Write the header of the file.
        f.write('SentenceID\tSentiment\n')

        # Write the predicted results to the file.
        for word, sentiment in zip(new_words, predictions):
            f.write(str(word.sentenceID) + '\t' + str(sentiment) + '\n')

class SentimentPredictor:
    """Predict sentiments for sentences using a Naive Bayes classifier."""
    
    def __init__(self, corpora):
        """Initialize the SentimentPredictor.

        Args:
            corpora (Train): An instance of the Train class containing the
                training data.
        """
        self.corpora = corpora
    
    def prior_probabilities(self):
        """Compute prior probabilities for each sentiment.

        Returns:
            list: A list of prior probabilities for each sentiment.
        """
        number_words_sentiment = self.corpora.number_words_sentiment

        # Use a list comprehension to compute the prior probabilities.
        return [
            number_word_sentiment / self.corpora.number_words
            for number_word_sentiment in number_words_sentiment
        ]

    def relative_likelihoods(self, word, sentiment_index, number_words_sentiment):
        """Compute the relative likelihood for a word for a given sentiment.

        Args:
            word (Word): The word to compute the likelihood for.
            sentiment_index (int): The index of the sentiment to compute the
                likelihood for.
            number_words_sentiment (list): A list of the number of words for
                each sentiment.

        Returns:
            list: A list of relative likelihoods for each token in the word.
        """
        rel_number_token_sentiments = self.corpora.rel_number_token_sentiments
        number_token_sentiments = self.corpora.number_token_sentiments
        number_vocab = self.corpora.number_vocab
        number_word_sentiment = number_words_sentiment[sentiment_index]

        # If there are no words with the given sentiment, return 0.
        if number_word_sentiment == 0:
            return 0

         # Get the relative token counts for the given sentiment.
        rel_token_number = rel_number_token_sentiments[sentiment_index]
        sentiment_number_of_tokens = number_token_sentiments[sentiment_index]
         # Use Laplace smoothing to prevent zero probabilities.
        smooth_sentiment = sentiment_number_of_tokens + number_vocab

        # Computes relative likelihoods for each token with smoothing.
        relative_likelihoods = [
            (1 + rel_token_number.get(token, 0)) / smooth_sentiment
            for token in word.tokens
        ]

        return relative_likelihoods

    def likelihoods(self, word, prior_probabilities):
        """Compute the likelihood of a word for each sentiment.

        Args:
            word (Word): The word to compute the likelihood for.
            prior_probabilities (list): A list of prior probabilities for each
                sentiment.

        Returns:
            list: A list of likelihoods for each sentiment.
        """
        likely = []

        # Iterate over each sentiment.
        for i in range(len(self.corpora.number_classes)):
            # Get the prior probability for the sentiment.
            prob = prior_probabilities[i]

            # Get the relative likelihoods for the word for the given sentiment.
            relative_likelihoods = self.relative_likelihoods(word, i, self.corpora.number_words_sentiment)

            # Compute the overall likelihood for the word by multiplying
            # the prior probability with the relative likely.
            for token in relative_likelihoods:
                prob *= token
            likely.append(prob)

        return likely

    def sentiments(self, words):
        """Predict the sentiments for a list of words.

        Args:
            words (list): A list of words to predict sentiments for.

        Returns:
            list: A list of predicted sentiments for the given words.
        """
        # Compute the prior probabilities.
        prior_probabilities = self.prior_probabilities()

        # Use a list comprehension to predict sentiments for each word.
        results = [
            # Determine the sentiment with the highest likelihood.
            max(range(len(likely)), key=lambda x: likely[x])
            for likely in (self.likelihoods(word, prior_probabilities) for word in words)
        ]

        return results

class Train:
    """Trains a model on the given words and number of classes."""

    def __init__(self, words, number_classes):
        """Initialize the training process.

        Args:
            words (list): List of words to train on.
            number_classes (int): Number of classes to train for.
        """
        self.number_classes = number_classes
        self.number_words = len(words)

        # Compute training data
        computed_data = self.train(words, number_classes)

        # Set attributes from computed data
        self.number_words_sentiment = computed_data["number_words_sentiment"]
        self.number_token_sentiments = computed_data["number_token_sentiments"]
        self.rel_number_token_sentiments = computed_data["rel_number_token_sentiments"]
        self.number_vocab = computed_data["number_vocab"]

    def train(self, words, number_classes):
        """Train the model on the given words and number of classes.

        Args:
            words (list): List of words to train on.
            number_classes (int): Number of classes to train for.

        Returns:
            dict: A dictionary containing the computed training data.
        """
        # Initialize empty lists to store computed data
        num_words = [0] * len(number_classes)
        number_tok = [0] * len(number_classes)
        number_relative = [dict() for _ in range(len(number_classes))]

        # Get sentiment and tokens for each word
        sentiment_tokens = [(word.sentiment, word.tokens) for word in words]

        # Iterate over each sentiment-token pair
        for sentiment, tokens in sentiment_tokens:
            # Increment the number of words with the given sentiment
            num_words[sentiment] += 1

            # Increment the total number of tokens for the given sentiment
            number_tok[sentiment] += len(tokens)

            # Count the number of occurrences of each token in the given sentiment
            for token in tokens:
                number_relative[sentiment][token] = number_relative[sentiment].get(token, 0) + 1

        # Compute the number of unique tokens in the given words
        number_vocab = len(set(token for word in words for token in word.tokens))

        # Return the computed training data
        return {
            "number_words_sentiment": num_words,
            "number_token_sentiments": number_tok,
            "rel_number_token_sentiments": number_relative,
            "number_vocab": number_vocab,
        }
    
