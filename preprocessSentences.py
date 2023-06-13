# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 2022

@author: Euan Goodbrand
"""

import string #for punctuation removal
import pandas as pd #for data manipulation
from nltk.corpus import stopwords #for stopword removal
from nltk.stem import PorterStemmer, WordNetLemmatizer #for stemming and lemmatization

class Preprocess:
    """Class to preprocess the words in the corpus.
    The preprocessing steps include:
    1. Converting tokens to lowercase
    2. Applying stemming and lemmatization to the tokens
    3. Removing stopwords
    """
    def __init__(self, words):
        """Constructor for Preprocess class.

        Args:
            words (list): A list of Word objects to be preprocessed.
        """
        self.words = words

        # Create instances of the PorterStemmer and WordNetLemmatizer classes
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize the stopwords set
        self.stopwords = set(stopwords.words('english') + list(string.punctuation.replace('!', '')) + [
                    '\'s', '``', '\'\'', '...', '--', 'n\'t', '\'d'])

        self.scaleClasses = [0, 0, 1, 2, 2]

    def preprocessWords(self):
        """Preprocess the words in the corpus.

        Returns:
            list: A list of preprocessed Word objects.
        """
        for word in self.words:
            # Convert tokens to lowercase
            lowercase_tokens = []
            for token in word.tokens:
                lowercase_token = token.lower()
                lowercase_tokens.append(lowercase_token)
            word.tokens = lowercase_tokens

            # Apply stemming and lemmatization to the tokens (removed as doesn't increase score)
            # stemmed_tokens = []
            # lemmatized_tokens = []
            # for token in word.tokens:
            #     stemmed_token = self.stemmer.stem(token)
            #     lemmatized_token = self.lemmatizer.lemmatize(token)
            #     stemmed_tokens.append(stemmed_token)
            #     lemmatized_tokens.append(lemmatized_token)
            # word.tokens = stemmed_tokens
            # word.tokens = lemmatized_token

            # Remove stopwords
            filtered_tokens = []
            for token in word.tokens:
                if token not in self.stopwords:
                    filtered_tokens.append(token)
            word.tokens = filtered_tokens
        return self.words

    def scaleDown(self):
        """Scale down the sentiment classes of the words in the corpus.

        Returns:
            list: A list of Word objects with scaled-down sentiment classes.
        """
        for word in self.words:
            old_sentiment = word.sentiment
            new_sentiment = self.scaleClasses[old_sentiment]
            word.sentiment = new_sentiment
        return self.words
    

class Sentence:
    """A class representing a single sentence with tokens and a sentiment."""

    def __init__(self, sentenceID, tokens, sentiment):
        """
        Initialize a new Sentence object.

        Args:
            sentenceID (int): The ID of the sentence.
            tokens (List[str]): A list of tokens in the sentence.
            sentiment (int): The sentiment of the sentence.
        """
        self.sentenceID = sentenceID
        self.tokens = tokens
        self.sentiment = sentiment

    def wordList(filename):
        """
        Extract a list of Sentence objects from a file.

        Args:
            filename (str): The name of the file to read from.

        Returns:
            List[Sentence]: A list of Sentence objects.
        """
        words = []
        df = pd.read_csv(filename, index_col=0, delimiter='\t')

        # Check if the Sentiment column is present in the dataframe.
        sentiment = 'Sentiment' in df.columns

        # Iterate over the rows in the dataframe.
        for index, row in df.iterrows():
            # Create a Sentence object with the word text and sentiment tag.
            word = Sentence(index, row['Phrase'].split(), 
                            row['Sentiment']) if sentiment else Sentence(index, row['Phrase'].split(), -1)
            words.append(word)

        return words