from nltk import download
from nltk.corpus import words
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.parse.corenlp import CoreNLPParser

# Download required NLTK corpora (if not already present)
download('punkt')
download('averaged_perceptron_tagger')
download("words")
download("stopwords")
download("wordnet")

# Load the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Retrieve the English dictionary
english_dictionary = set(words.words())

parser = CoreNLPParser()

# TODO: non-temporal since and non-comparison as

# Define premise conclusion markers as in
# https://academic.csuohio.edu/polen/LC9_Help/1/11pcindicators.htm
premise_conclusion_markers = ['since', 'as indicated by', 'because', 'for', 'in that', 'as', 'may be inferred from', 'given that', 'seeing that', 'for the reason that', 'inasmuch as', 'owing to', 'therefore', 'wherefore', 'accordingly', 'we may conclude', 'entails that', 'hence', 'thus', 'consequently'] 