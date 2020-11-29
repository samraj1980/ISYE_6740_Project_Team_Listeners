# Listen to our representatives: Using Topic Modeling and Topic Classification to understand the message themes from Congress during the 2020 election.



[![](https://img.shields.io/badge/authors-%40Neepa%20Biswas-blue)](https://www.linkedin.com/in/neepa-biswas-3325b87/)
[![](https://img.shields.io/badge/authors-%40Sam%20Raj-blue)](https://www.linkedin.com/in/sam-raj-anand-jeyachandran-pmp-7b273a6/)
[![](https://img.shields.io/badge/authors-%40Alain%20R.Garcia-blue)](https://www.linkedin.com/in/renegar)


## 1 Problem statement  
In a heavily contested US presidential race, we want to use Machine Learning to understand what the main themes are in each party broadcasted through Twitter. We will focus on tweets from the house of representatives, across the Republican and Democratic parties. 
We want to understand what key themes are being pushed out by each party, how consistent are the messages within the parties and what is the reach (geo). In general, we want to understand how cohesive each campaign is in regards to their messages.  
In the process we want to also understand which is the best Natural Language Processing (NLP) algorithm for this problem, and what are the pros/cons of each algorithm. LDA (Latent Dirichlet Allocation), LSA or LSI – Latent Semantic Analysis or Latent Semantic Indexing and NMF (Non-Negative Matrix Factorization) are the three common NLP algorithms that are used in topic modeling.  

## 2 Data Source
The data set was sourced from Twitter. It consists of a data stream with tweets from the members of the house of representatives (435 members) along with additional metadata captured from the harnessed tweets. We aimed to keep the information unbiased by having an equal number of representatives across party lines. This data was gathered using a twitter ingestion module. The training data corresponds to tweets made from Jan-2019 to mid Oct-2020, representing a total of 101,799 tweets. Of the data only the Democratic and Republican affiliations were used for the project (99402 Tweets in Total). Re-tweets were excluded as well as tweets in any language other than English. 

### 2.1. Dataset exploratory analysis
 
<table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture1.png">
    </td>
  </tr>
</table> 
 
In Fig 1. We see the total volume of tweets growing over time, as we get closer to the election day. Overall Democrats had consistently double of the volume of tweets than republicans.

<table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture2.png">
    </td>
  </tr>
</table> 

As we can see in Fig 2 there is a group of 28 representatives that represent the 48% of the total volume of tweets. Across party lines, the results are very similar, with the top 20 republican representatives representing 62% of the total volume, while 59% for the top 20 democratic representatives. 
 
 <table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture3.png">
    </td>
  </tr>
</table> 
 
Twitter use of representatives by state is more prevalent in California, followed by New Jersey and Illinois. The share of tweets in these 3 states has a heavy Democrat bias. While in states like Ohio and Florida, the volume is well balanced across party lines. 

## 3 Methodology 
At a high level, our process was to pre-process the data (tweets) procured by the ingestion module and then create separate topic models for democrats and republicans using three commonly used NLP algorithms: LDA, MNF & LSA, using Coherence score as the evaluation metric across all models. Then we combined all 3 models with an ensemble process to assign the best topic to each tweet. The result was then analyzed through data visualization. 
As an extra step, we also created a classification model using the ensemble result to predict topic for tweets in the future, this was done using a Bayes classifier and tested with a stream of tweets dated after our training data was captured.  
All coding was done in Python using Jupiter notebooks with GenSim, SKLearn, tweepy , nltk , scapy among other libraries.
Below is a flowchart of the main process:

Fig 4: Flow chart of the main process 

 <table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture4.png">
    </td>
  </tr>
</table> 

## 4 Data Preprocessing
In topic modeling, data preparation is critical to end with meaningful topics, especially since it’s an unsupervised form of learning. Optimization happens in the data preprocessing as well in the algorithm tuning. Below is the final sequence of preprocessing after optimization:
•	Read the tweets from the staging Azure DB.
•	Removed all punctuations.
•	Removed all URLs.
•	Removed commonly used words (stopwords) for both English and Spanish.
•	Performed Lemmatization to keep only Nouns and Adjectives.
•	Remove additional words (not relevant or ambiguous) based on analyzing remaining word clouds.
•	Replaced synonyms.


### 4.1 Final word cloud result after optimization of non-relevant or conflicting words:
Fig 5: Final word count

 <table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture5.png">
    </td>
  </tr>
</table>
 

### 4.2 Top 20 most common two-word pairs (Two Gram): 
Fig 6: Top 20 most common two-word pair (bi-grams)

 <table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture6.png">
    </td>
  </tr>
</table>


## 5 Evaluation
For each algorithm ran on the common Document Term Matrix (DTM) that resulted after building a bag-of-word on the cleaned data set describe in step 4. To evaluate the best set of hyperparameters for each algorithm we selected the Coherence score. Coherence  calculates if the words in the same topic make sense when they are put together. Coherence approximates Human Word Intrusion very well, as opposed to other metrics like Perplexity and Log Likelihood which can correlate negatively with Human Word Intrusion. These measurements help distinguish between topics that are semantically interpretable topics and topics that are artifacts of statistical inference. We settled on Cv as the Coherence measure due to it being generally the better performing, although it was more resource intensive, than Umass. Different approaches were taken to determine the optimal model for the three algorithms:

### 5.1 LDA Evaluation 
For LDA, 3 different model variations were tested for coherence.
1.	Plain LDA model iterated for the number of topics (5 to 35)
2.	The LDA model was iterated using a combination of the following hyperparameters: Number of Topics (5 to 35), document topic density alpha (range 0.01 to 1, symmetric and asymmetric) and topic word density beta (range 0.01 to 1 and symmetric).
3.	The model was iterated using a combination of the following hyperparameters: Number of Topics (5 to 35), Learning Decay (0.5, 0.7 & 0.9), document topic density alpha (0.01) and topic word density beta (0.01). 
Model 3 had a higher coherence score than the other two models, with highest coherence score at 0.55 with 16 Topics and Learning Decay 0.9.
 
 <table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture7.png">
    </td>
  </tr>
</table>


### 5.2 NMF Evaluation 
For NMF, 3 different model variations were tested for coherence, across a range of topics from 5 to 35: 
1.	Plain NMF, using Coordinate Descent solver and Frobenius norm and no regularization.
2.	Modified NMF, same as above with regularization of alpha=.1 and regularization mixing parameter l1_ratio=.5
3.	Probabilistic Latent Semantic Indexing: NMF using Kullback-Leibler divergence and same regularization as in 2



The modified NMF had a higher Coherence score than the other 2 models, with the highest coherence peaking at 17 topics (0.58).
Fig 8: Results from NMF model

 <table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture8.png">
    </td>
  </tr>
</table>

### 5.3 LSA Evaluation 
LSA (truncated SVD) being the simples of all 3 algorithms, has no other hyperparameters than the number of topics. The coherence score was evaluated variating topics from 5 to 35.
The highest coherence score was 0.33 with 20 topics, considerably lower than the previous two algorithms. 
Fig 9: Results from LSA model
 
 <table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture9.png">
    </td>
  </tr>
</table>


### 5.4 Summary
Overall, both NMF and LDA had the best coherence score, while LSA had the lowest. Computationally LDA was more resource intensive than NMF, while again producing similar results in this case. This is something to keep in mind, as extending this analysis to broader base of twitter users would expand the data set considerably. 

## 6 Final Ensemble
To label each tweet, the following four steps were performed. 
1.	The top three occurring words produced by each of the three topic- modeling algorithms for this tweet were merged as a string of nine words.  
2.	Tweets for which the topic probability produced by the NMF and LSA models were zero, were ignored since these algorithms did not have a good confidence on the topic modeled. This reduced the total amount of tweets from 99K to 53K. 
3.	Then, the 2 highest occurrence words from this combined string of words were selected. 
4.	Finally, the two high frequency words selected from previous step, were normalized by scanning against the list of two-gram topics that were created during exploratory analysis. The two-gram topic which matches the closest was picked as the final topic name for that tweet. 

 <table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture10.png">
    </td>
  </tr>
</table> 


Figure 10: Sample Ensemble Data

 <table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture11.png">
    </td>
  </tr>
</table>



## 7 Results
Fig 11 shows the 17 topics generated by the three models after the ensemble process.

 <table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture12.png">
    </td>
  </tr>
</table>
 

Of the resulting data set of around 55K tweets, here were the top eight topics (out of 17 total) broken out by party (sorted by tweet volume). The top eight topics represent 80% of the tweet volume for each party. 

 <table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture19.png">
    </td>
  </tr>
</table>


### 7.1 How consistent were the messages throughout the campaign? 
For Democrats (Fig 12) the topic of Health Care dominated consistently throughout, followed by Job Economy, these two represented anywhere from 28% to 52% of the share on any given month. Most of the other topics were very consistent as well, but there were some seasonal topics as well like COVID-19 which took away share towards the end of the election cycle.  

 <table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture13.png">
    </td>
  </tr>
</table> 

For Republicans (Fig 13) the topic of Job & Economy followed by Veteran & Service, were the two dominating topics. These also represented anywhere from 23% to 52% of the share in any given month. Here we saw a bit more influence by seasonal topics like Border Security being replaced by the Impeachment process and then subsequently replaced by Small Business.  

 <table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture14.png">
    </td>
  </tr>
</table>
 
### 7.2 How consistent were the messages by State? 
Overall the Job/Economy topic was the leading topic in most states, followed by Covid-19 (which is very heavily related to the Economic topic). 

 <table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture15.png">
    </td>
  </tr>
</table>
 
## 8 Predicting future Topics
As an extra goal, we wanted to use the training set and build a model to predict topics for tweets that happened after our training data scope. We selected a Naïve Bayes Classifier. Naïve Bayes is based on probability of events and performs well in text classification, as it uses Laplace smoothing. It also requires less training data and training time. We then re-ran our ingestion model to capture tweets post Mid October (test set, around 7700 tweets) and pre-processed the data with the same process we had for our training set. Finally, we trained the model using the word count and the final normalized topic for the training set and predicted the topic for all the new tweets in the test set. 

We randomly selected a group of tweets and compared the predicted topic vs our interpretation, we found that this classifier performed very well overall. The results from 20 random samples show an accuracy of 90%. Below is a sampling of tweets and their predicted topic: 

 <table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture16.png">
    </td>
  </tr>
</table>
 

For the 7K test tweets (Fig 15), we see the democrats continued to tweet more volume of tweets than republicans. The main themes continued, around Job/Economy following and Health Care, but Vote bumping up to second place in terms of volume for democrats and third place for republicans. This was expected considering the test data was taken from mid-October to mid-November during which election was held. 

 <table>
  <tr>
    <td>
      <img src="https://github.com/samraj1980/ISYE_6740_Project/blob/main/Picture17.png">
    </td>
  </tr>
</table>

## 9 Conclusions
In this project, we investigated topic modeling from tweets from the members of the house of representatives to see the trending topics among Democrats and Republicans using LDA, LSA and NMF models. Based on the results from the three algorithms an ensemble was created to determine the top trending topics. It was found the top trending topics among Democrats were health care followed by job/economy, whereas among Republicans the top two topics were job/economy and veterans/service. A topic classification model using Naïve Bayes was also built using the ensemble results to predict the topic for future tweets (test data). Results from random samples showed 90% accuracy.  Vote/Election was an important topic in the test data as the time frame of this data coincided with the 2020 Presidential election. 

## 10 References
1.	Comparing Twitter and Traditional Media Using Topic Models, European Conference on Information RetrievalECIR 2011: Advances in Information Retrieval pp 338-349, Wayne Xin ZhaoJing Jiang, Jianshu Weng, Jing He, Ee-Peng Lim, Hongfei Yan, Xiaoming Li
2.	Latent Dirichlet Allocation David M. Blei, Andrew Y. Ng, Michael I. Jordan; 3(Jan):993-1022, 2003.
3.	Probabilistic author-topic models for information discovery::M Steyvers, P Smyth, M Rosen-Zvi… - Publication: KDD '04: Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining. August 2004 Pages 306–315https://doi.org/10.1145/1014052.1014087
4.	Empirical Study of Topic Modeling in Twitter :Liangjie Hong and Brian D. Davison Publication: SOMA '10: Proceedings of the First Workshop on Social Media Analytics July 2010 Pages 80–88 https://doi.org/10.1145/1964858.1964870
5.	https://nlp.stanford.edu/IR-book/html/htmledition/latent-semantic-indexing-1.html
6.	An Introduction to Latent Semantic Analysis :Thomas K Landauer Department of Psychology University of Colorado at Boulder, Peter W. Foltz Department of Psychology New Mexico State University Darrell Laham Department of Psychology University of Colorado at Boulder, Landauer, T. K., Foltz, P. W., & Laham, D. (1998). Introduction to Latent Semantic Analysis. Discourse Processes, 25, 259-284.
7.	https://www.analyticssteps.com/blogs/introduction-latent-semantic-analysis-lsa-and-latent-dirichlet-allocation-lda
8.	Nonnegative matrix factorization for interactive topic modeling and document clustering Da Kuang and Jaegul Choo and Haesun Park
9.	Spatial Aggregation Facilitates Discovery of Spatial Topics :Aniruddha Maiti Temple University Philadelphia, PA-19122, USA aniruddha.maiti@temple.edu Slobodan Vucetic Temple University Philadelphia, PA-19122, USA vucetic@temple.edu
10.	Performance Analysis of Topic Modeling Algorithms for news articles: T.Rajasundari1, P.Subathra2,P.N.Kumar3. Amrita University, India

11 Work Distribution
Neepa Biswas	LDA Model Implementation & Tuning, Bayes Model Tuning, Data Preprocessing Tuning, Exploratory Data Analysis, Project write up
Samraj A. Jeyachandran	LSA Model Implementation & Tuning, Data Preprocessing Tuning, Exploratory Data Analysis, Tweeter Ingestion Module, Ensemble Module Implementation and Tuning
Alain R. Garcia	NMF Model Implementation & Tuning, Exploratory Data Analysis, Results Analysis & Visualizations, Data Pre-processing Tuning, Project write up 

