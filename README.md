# Linguistic Cues Unveiling Fake News (ADA extension project)

## Abstract

Our starting point is the paper [Linguistic Harbingers of Betrayal: A Case Study on an Online Strategy Game](https://arxiv.org/abs/1506.04744). It revealed the subtle signs of imminent betrayal encoded in the conversational patterns of dyads of players in the game Diplomacy. 

Our intention is to explore whether the linguistic cues that have been found in the original paper are suitable for a descriptive/predictive analysis in a different but still related context. The theme we want to discuss is "fake news". Since the advent of social networks, fake news has become a major issue. Traditionally, information has always been filtered and verified before publication. These new media give accessibility to information spreading to anyone that has an Internet connection. 

Starting from a dataset of plain text labeled news, our goal is thus to compute some of the linguistic features individuated in the original paper (such as argumentation, talkativeness, etc.) and to discover further indicators if needed, then build a machine learning model that will possibly distinguish between fake and real news.

## Research questions 

- Is there any linguistic cue that clearly identifies a fake news?
- Are the linguistic features mentioned in the paper suitable for an extension of the analysis to a domain different from that of communication between players in board game? 
- If not entirely, which could the other indicators be?

## Proposed datasets

[Fake News detection](https://www.kaggle.com/jruvika/fake-news-detection)

This dataset has been downloaded from Kaggle. It consists of 4009 distinct real and fake labeled news. Each news has four attributes: *URLs*, *Headline*, *Body* and *Label*. Although, note that we do not plan to use the *URLs* feature, since we want to base our analysis on pure linguistic cues. It would be an easy task to classify such news based on the URL of the website that published it, since some websites are a priori trustworthy and others are not.    

## Methods 

#### Data collection and preprocessing:

1. Load the dataset

2. Extract features from the plain text articles

   a. Extract the sentiment score extration with both the tools BERT and NLTK
   
   b. Extract the *subjectivity* using [TextBlob Sentiment Analizer](https://planspace.org/20150607-textblob_sentiment/)

   c. Calculate the number of sentences and the average number of words per sentence from each article to represent the *talkativeness*

   d. Extract the number of *discourse markers* from the articles

      * Prepare Stanford parse trees from the articles
      * Extract the discourse markers from the parse trees using [this](https://github.com/erzaliator/DiscourseMarker) discourse marker highlighter

   e. Count the number of *planning markers* in the articles

      * Perform P.O.S. Tagging with [Penn Treebank](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
      * Parse the resulting tree and count the number of verbs in future tense

   f. Count the number of claim and premise markers with [this](https://academic.csuohio.edu/polen/LC9_Help/1/11pcindicators.htm) lexicon
   g. Count the number of grammatical mistakes

#### Model setup:

 1. Test the original model used in the paper to perform binary classification on the extracted features and the authenticity of the news. The model used here is logistic regression with univariate feature selection and 5-fold cross-validation.
 2. Test SVD performances.

##### Data Analysis and Visualization:

You can find a presentation of our results in report.pdf or directly give a look to the Jupyter notebook paper_extension.ipynb

## Repository Structure

Our repository has the following structure:
```bash
├── README.md
|
├── data
│   ├── data.csv
│   ├── data_bertsent.csv
│   ├── discourseMarkers.data
│   └── reviews.csv
|
├── coreNLPServer.py
├── discourseMarkers.py
├── paper_extension.ipynb
├── resources.py
└── sentiment_bert.py
└── report.pdf
```

* data/data.csv contains the original dataset.
* data/data_bertsent contains the sentiment score from our SentimentBERT classifier.
* data/discourseMarkers.data contains lists of discourse markers for each articles.
* data/reviews.csv contains the dataset that SentimentBERT is trained on.

* coreNLPServer.py is the script to run the Standford CoreNLP server. You need to [download the server](https://stanfordnlp.github.io/CoreNLP/download.html) befor running it
* discourseMarkers.py is used to create discourseMarkers.data file.
* resources.py contains the required NLTK tools and instances of external classes, for example, list of premise markers.
* sentiment_bert.py contains the SentimentBert class and training scripts for our SentimentBERT classifier.
* **paper_extension.ipynb** contains the results of our reproduction. It should be used as the main file for evaluation.
* **report.pdf** is the summary of our project.
