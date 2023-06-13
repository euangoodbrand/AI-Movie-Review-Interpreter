# Sentiment Analysis

This project is a Naïve Bayesian sentiment analysis model designed to analyze movie reviews. The implementation consists of several files, with the main file being `NB_sentiment_analyser.py`. Here's an overview of the project's components:

- `preprocessSentences.py`: This file handles the preprocessing of sentences by removing punctuation, stop words, and performing lemmatization and stemming. It also rescales the data based on the required number of classes, converting a five-class sentiment system to a three-class sentiment system.

- `featureExtraction.py`: This file extracts features by calculating a sentiment score for each feature token and sorting them in a list. A certain percentage of the highest-scoring features is selected based on the specified `featurePercentage`. The sentiment scores are adjusted to match the sentiment scale of the desired classes.

- `naïveBayes.py`: This file trains the corpus using a Naïve Bayes model on a labeled dataset.

- `evaluate.py`: This file evaluates the predictions using Macro F1 scores and plots a confusion matrix.

## Evaluation Results

The evaluation results were analyzed in two graphs:

1. Macro F1 Scores with Preprocessing: This graph shows the Macro F1 scores achieved by applying different preprocessing steps, such as lowercase conversion, stop-word removal, stemming, and lemmatization. The results vary for the three-class and five-class models, with preprocessing consistently improving the three-class model but negatively impacting the five-class model. Lowercase processing and stop-word removal yielded the most significant improvements for the three-class model, while the best results were obtained with a combination of lowercase processing and stop-word removal. Lemmatization and stemming did not contribute further improvements. In contrast, the five-class model achieved the best results without any preprocessing.

2. Macro F1 Scores with Feature Extraction: This graph compares the Macro F1 scores obtained with feature extraction versus using all words as features. Feature extraction consistently outperforms using all features, with the five-class model showing more significant improvements compared to the three-class model. The three-class model achieved its best results using 97% of the features, which is similar to using all the features. However, the five-class model greatly benefited from feature extraction, with its best-performing feature amount being 38%, a substantial improvement over using all words as features.

## Setup and Running the Code

To run the code, ensure you have Python 3.9.x or above installed, along with the following libraries: numpy, pandas, collections, string, re, seaborn, matplotlib, and nltk. If you're using the Anaconda distribution of Python, most of these libraries should already be included.

To install the NLTK library, execute the following commands in the Anaconda prompt:

pip install nltk
import nltk
nltk.download('all')


After downloading the code, navigate to the correct directory and run the following command:

python NB_sentiment_analyser.py moviereviews/train.tsv moviereviews/dev.tsv moviereviews/test.tsv -classes <3 or 5> -features <all_words or features> -confusion_matrix


Make sure to replace `<TRAINING_FILE>`, `<DEV_FILE>`, and `<TEST_FILE>` with the paths to the corresponding training, development, and test files. Additionally, specify the number of classes using `-classes <NUMBER_CLASSES>` (either 3 or 5), choose between using selected features or all words with `-features`, and add the `-confusion_matrix` parameter to show confusion matrices (optional).

Feel free to explore the code and experiment with different parameters to fine-tune the sentiment analysis model.

