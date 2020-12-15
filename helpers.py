import pandas as pd
import numpy as np
import math
import nltk
import re
from nltk import word_tokenize
from nltk import pos_tag
from nltk.tokenize import sent_tokenize
from resources import lemmatizer, premise_conclusion_markers, english_dictionary, wordnet, parser


def invert_quotes_fullstop(sentence):
    '''
    Substitute the expression '."' with '".' -> done to help sentence tokenization
    '''
    return sentence.replace('.”', '”.')


def split_sentences(article):
    sentences = []
    for period in article.split("\n"):
        for sentence in sent_tokenize(period):
            sentences.append(sentence)
    return sentences


def count_future_verbs(sentence):        
    text = word_tokenize(sentence)
    tagged = pos_tag(text)
    return len([word for word in tagged if word[1] in ["VBC", "VBF"]])


def count_premise_conclusion_markers(sentence):        
    counter = 0
    for marker in premise_conclusion_markers:
        if sentence.find(marker) != -1:
            counter += 1
    return counter


def get_wordnet_pos(treebank_tag):
    '''
    Helper function to convert a POS tag from the Penn Treebank to a WordNet format
    :param treebank_tag: Penn Treebank POS tag, string
    :returns: WordNet POS tag
    '''

    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_words(words):
    '''
    Function to lemmatize word tokens using the WordNet lemmatizer
    :param words: list of word tokens in the tweet
    :returns: list of lemmatized word tokens
    '''

    lemmatized_words = []
    # Get the Penn Treebank POS tags for the word tokens
    word_pos_tags = pos_tag(words)
    for word, word_pos_tag in word_pos_tags:
        # Get the WordNet POS tag
        word_pos_tag = get_wordnet_pos(word_pos_tag)
        # Use the WordNet POS tag to lemmatize the word into the correct word form
        lemmatized_words.append(lemmatizer.lemmatize(word, word_pos_tag))
    return lemmatized_words


def get_word_tokens(sentence):
    '''
    Function to retrieve only tokens corresponding to real English words
    :param sentence: string corresponding to an English sentence
    :returns: parsed list of tokens with only word tokens
    '''
    tokens = word_tokenize(sentence)
    return [token for token in tokens if re.match(r"^[A-Za-z\']+$", token)]


def parse_tree(sentence):
    ''' Create the parse tree for the given sentence '''
    return next(parser.raw_parse(sentence))