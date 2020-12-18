# ada-2020-project-milestone-p3-p3_thot



#### 1. Title

~~Linguistic Harbinger of Fake News~~

Linguistic Cues Unveiling Fake News

#### 2. Abstract:

The original paper revealed the subtle signs of imminent betrayal encoded in the conversational patterns of a dyad of players in the game Diplomacy. 

Our intention is to explore whether the linguistic cues that have been found in the original paper are suitable for a descriptive/predictive analysis in a different but still related context. The theme we want to discuss is "fake news". Since the advent of social networks, fake news has become a major issue. Traditionally, information has always been filtered and verified before publication. These new media give accessibility to information spreading to anyone that has an Internet connection. 

Starting from a dataset of plain text labeled news, our goal is thus to compute some of the linguistic features individuated in the original paper (such as argumentation, talkativeness, etc.) and to discover further indicators if needed, then build a machine learning model that will possibly distinguish between fake and real news.

#### 3. Research questions 

- Is there any linguistic cue that clearly identifies a fake news?
- Are the linguistic features mentioned in the paper suitable for an extension of the analysis to a domain different from that of communication between players in board game? 
- If not entirely, which could the other indicators be?

#### 4. Proposed datasets

[Fake News detection](https://www.kaggle.com/jruvika/fake-news-detection)

This dataset has been downloaded from Kaggle. It consists of 4009 distinct real and fake labeled news. Each news has four attributes: *URLs*, *Headline*, *Body* and *Label*. Although, note that we do not plan to use the *URLs* feature, since we want to base our analysis on pure linguistic cues. It would be an easy task to classify such news based on the URL of the website that published it, since some websites are a priori trustworthy and others are not.    

#### 5. Methods 

##### Data collection and preprocessing:

1. Load the dataset

~~2. Append the body column to the headline column, to form a complete article.~~

3. Extract features from the combined articles

   ~~1. Extract the politeness level from the article with [Stanford Politeness classifier](https://github.com/sudhof/politeness/tree/python3)~~

      ~~* Extract the articles from the dataset and prepare them in txt form
      ~~* Pipeline the text documents to Standford CoreNLP and collect their dependency parsings
      ~~* Prepare a list of dictionaries with articles and parses as keys and their content as values
      ~~* Send the dictionary to the classifier and obtain the politeness score

   2. Extract the *subjectivity* using [TextBlob Sentiment Analizer](https://planspace.org/20150607-textblob_sentiment/)

   3. Calculate the number of sentences and the average number of words per sentence from each article to represent the *talkativeness*

   4. Extract the number of *discourse markers* from the articles

      * Prepare Stanford parse trees from the articles
      * Extract the discourse markers from the parse trees using [this](https://github.com/erzaliator/DiscourseMarker) discourse marker highlighter

   5. Count the number of *planning markers* in the articles

      (Alternative 1)

      * Design a rough future temporal marker lexicon
      * Count the number of occurrences of the markers in the document

      (Alternative 2)

      ~~* Perform P.O.S. Tagging with [Penn Treebank](https://www.google.com/search?q=penn+treebank&oq=penn+treebank&aqs=chrome..69i57j0j0i20i263j0l5.5027j0j4&sourceid=chrome&ie=UTF-8)~~
      * Perform P.O.S. Tagging with [Penn Treebank](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
      * Parse the resulting tree and count the number of verbs in future tense

   6. Count the number of claim and premise markers with [this](https://academic.csuohio.edu/polen/LC9_Help/1/11pcindicators.htm) lexicon

   7. (Optional) Extract more linguistic features from the articles (e.g. TF-IDF, number of grammatical mistakes, etc.)

##### Model setup:

 1. Test the original model used in the paper to perform binary classification on the extracted features and the authenticity of the news. The model used here is logistic regression with univariate feature selection and 5-fold cross-validation.
 2. If the original one fails in explaining the data, we can explore more complex models, e.g. random forest or gradient boosting. This will let us check whether the unsatisfactory result comes from the model simplicity or from the fact that the extracted linguistic cues are not significant enough to predict fake news with high accuracy.

##### Data Analysis and Visualization:

 1. Summarize the model statistics, visualize the key findings and discuss the results.
 2. Organize the analysis and write a data story about it on an interactive web page.
 3. Prepare the video presentation.

#### 6. Proposed timeline

- Week 0.5: Prepare and familiarize with the tools needed for feature extraction.
- Week 0.5 - Week 1.5: Feature extraction.
- Week 1.5 - Week 2: Modeling and data analysis.
- Week 3: Preparing the web page data story and video presentation.

#### 7. Organization within the team:

Tianzong:

- will start the feature engineering by applying methods 3.1 - 3.3;
- will try to conceive, together with Orazio, the features to extract for task 3.7;
- will build the logistic regression model and train it on the dataset;
- will write the textual descriptions of the data story.

Orazio:

- will start the feature engineering by applying methods 3.4 - 3.6;
- will try to conceive, together with Tianzong, the features to extract for task 3.7;
- will build another model (undecided yet) and train it on the dataset;
- will create any interactive or static plot needed in the data story.

Tristan:

- will compare the results and choose the best model;
- will put the textual and visual data story together on a web page;
- will prepare a video presentation.




# Update

### Repository Structure

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

###  Methods

* The project title was changed.
* Article headlines was discarded from our model.
* The original step 3.1 was replaced with sentiment score extration wih BERT and NLTK.
* The link of Penn Treebank was updated.


### Contributions of Group Members

Tianzong:

- Feature extration 3.1 - 3.3;
- Logistic regression & SVM modelling;
- Result visulization and discussion;
- Writing report;

Orazio:

- Feature extration 3.4 - 3.6;
- Feature extraction 3.7 by adding *dictionary percentage* feature;
- Explorotory data analysis on extracted features and visulizations;
- Writing report;
- Will prepare a video presentation.



