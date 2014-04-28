#!/usr/bin/python

from bs4 import BeautifulSoup
from gensim import corpora, models
from gensim.models import ldamodel
import math
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import numpy
import regex
from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans, SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GMM
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
from sklearn import metrics
from textblob import TextBlob, Word
from textblob.classifiers import NaiveBayesClassifier

def main():
  # Whether we are using scilearn for features or our own custom.
  scilearn = True

  soups = []
  raw_texts = []
  texts = []
  topics = []
  training_set = []
  training_data = []
  training_topics = []
  test_set = []
  test_data = []
  test_topics = []

  interested_topics = {'corn', 'earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat'}

  for i in range(0, 22):
    num = str(i)
    if (i < 10): 
      num = "0" + str(i);
    #if i == 0 or i == 21:
    if True:
      new_soup = BeautifulSoup(open("/home/rawr/uni/data_mining/ass/data/reut2-0"+num+".sgm"))
      soups.append(new_soup)

  print 'loaded %d soups' % len(soups)
    
  for i in range(0, len(soups)):
    this_soup = soups[i]
    these_bodies = this_soup.find_all('text')
    reuters = this_soup.find_all('reuters')
    lewis = []
    these_topics = []
    for reut in reuters:
      lewis.append(reut.get('lewissplit'))
      these_topics.append(reut.topics.findChildren())
    
    for j in range(1, len(these_topics)):
      for topic in these_topics[j]:
        topic = str(topic)[3:-4]
        if topic in interested_topics:
          body = these_bodies[j].text
          if lewis[j] == 'TRAIN':
            training_set.append([clean_body(body, scilearn), topic])
            training_data.append(clean_body(body, scilearn))
            training_topics.append(topic)
          elif lewis[j] == 'TEST':
            test_set.append([clean_body(body, scilearn), topic])
            test_data.append(clean_body(body, scilearn))
            test_topics.append(topic)
          if lewis[j] == 'TRAIN' or lewis[j] == 'TEST':
            texts.append(clean_body(body, scilearn))
            raw_texts.append(body)
            #print clean_body(body, scilearn)
            topics.append(topic)

  topic_dict = {}
  for topic in interested_topics:
    topic_dict[topic] =  topics.count(topic)
    print topic + ': ' + str(topics.count(topic))
  print 'total number of docs: %d' % len(topics)
  print '***********************'

  # FOR TESTING
  if len(test_data) == 0:
    print 'Making artificial test data'
    text = 'i am a test data, hello!'
    topic = 'earn'
    test_set.append([clean_body(body, scilearn), topic])
    test_data.append(clean_body(body, scilearn))
    test_topics.append(topic)

  if scilearn:
    vect = TfidfVectorizer(strip_accents = 'unicode',
                           sublinear_tf = True,
                           max_df = 0.5,
                           stop_words = 'english')
    featured_train = vect.fit_transform(training_data)
    print 'Training: n_samples: %d, n_features: %d' % featured_train.shape
    featured_test = vect.transform(test_data)
    print 'Test: n_samples: %d, n_features: %d' % featured_test.shape
    featured_texts = vect.fit_transform(texts)
    print 'Fininshed TfIdf vectoriser'

    chi = SelectKBest(chi2, 10)
    featured_train = chi.fit_transform(featured_train, training_topics)
    featured_test = chi.transform(featured_test)
    print 'Finished chi^2'

    feature_names = numpy.asarray(vect.get_feature_names())
    print feature_names

    classify(MultinomialNB(alpha=0.1),
             featured_train,
             training_topics,
             featured_test,
             test_topics,
             interested_topics)
             #True)

    classify(LinearSVC(),
             featured_train,
             training_topics,
             featured_test,
             test_topics,
             interested_topics)
             #True)

    classify(RandomForestClassifier(),
             featured_train.toarray(),
             training_topics,
             featured_test.toarray(),
             test_topics,
             interested_topics)
    # (supervised)
    classify(KNeighborsClassifier(),
             featured_train,
             training_topics,
             featured_test,
             test_topics,
             interested_topics)

    classify(NearestCentroid(),
             featured_train,
             training_topics,
             featured_test,
             test_topics,
             interested_topics)

    # **** Clustering ****
    print '****************** Clustering ******************'
    num_clusters = len(interested_topics)
    cluster(KMeans(n_clusters=num_clusters), featured_texts, topics)
    cluster(SpectralClustering(n_clusters=num_clusters), featured_texts, topics)
    cluster(AffinityPropagation(), featured_texts, topics)
    #cluster(DBSCAN(), featured_texts, topics)
    # GMM
    gmm = GMM(n_components=num_clusters)
    gmm.means_ = numpy.array([featured_texts[topics == i].mean(axis=0) for i in xrange(num_clusters)])
    #cluster(gmm, featured_texts, topics)

  if not scilearn:
    # Feature selection methods:
    #bag_of_words(texts)
    #print calc_tf_idf(texts, 10)

    print 'Alternative Naive Bayes:'
    # Classification
    NB = NaiveBayesClassifier(training_set)
    print test_set[0]
    print NB.classify((test_set[0])[0])
    print test_set[(test_set[0])[1]]
    #print NB.classify(['things', 'profit'])

    #print NB.classify(training_set[0][0])
    #print training_set[0][1]

    print NB.accuracy(test_set)
    print NB.show_informative_features(50)


def classify(classifier,
             training_data,
             training_topics,
             test_data,
             test_topics,
             topics,
             report=True):
  print 'training: ' + str(classifier)
  classifier.fit(training_data, training_topics)
  print 'testing'
  pred = classifier.predict(test_data)
  score = metrics.f1_score(test_topics, pred)
  print 'f1 score: %f' % score
  print 'report:'
  if report:
    print metrics.classification_report(test_topics, pred)


def cluster(classifier, data, topics):
    print str(classifier)
    classifier.fit(data)
    labels = classifier.labels_
    print 'Homogeneity: %0.3f' % metrics.homogeneity_score(topics, labels)
    print 'Completeness: %0.3f' % metrics.completeness_score(topics, labels)
    print 'Silhouette test: %0.3f' % metrics.silhouette_score(data, labels)
    print ' ***************** '


def calc_tf_idf(texts, x):
  all_scores = {}
  for i, text in enumerate(texts):
    scores = {word: tfidf(word, text, texts) for word in text.words}
    # Sort words by tf-idf 
    sorted_words = sorted(scores.items(), key = lambda x: x[1], reverse = True)
    # Take top x words and store in all_scores
    all_scores[text] = sorted_words[:x]
  return all_scores


def tf(word, text):
  return float(text.words.count(word)) / float(len(text.words))


def num_texts(word, texts):
  return sum(1 for text in texts if word in text)


def idf(word, texts):
  return math.log(float(len(texts)) / float(1 + num_texts(word, texts)))


def tfidf(word, text, texts):
  return tf(word, text) * idf(word, texts)


def bag_of_words(texts):
  # Use dictionary to create 'Bag of words'
  dictionary = corpora.Dictionary([i.words for i in texts])

  # Apply thresholding to reduce dimensionality - remove all
  # words which appear only once over all documents
  once_ids = [i for i, j in dictionary.dfs.iteritems() if j == 1]
  dictionary.filter_tokens(once_ids)
  # Remove words that appeared in less than 3 documents
  dictionary.filter_extremes(no_below=3)
  dictionary.compactify()

  # Convert the bodies to sparse vectors
  vectors = []
  for text in texts:
    vectors.append(dictionary.doc2bow(text.words))

  # Apply an LDA Model to the bag of words
  model = ldamodel.LdaModel(vectors, id2word=dictionary, num_topics=10)
  # For example print the probability distribution for the first text
  print model[vectors[0]]
  print model[vectors[1]]
  print model[vectors[2]]


# Method to carry out pre-processing and cleaning of bodies
def clean_body(body, scilearn):
  # If using scilearn for feature extraction just remove whitespace
  if scilearn:
    return ' '.join(body.split())
  # Remove title and date - we only want the text. Join also removes excess whitespace
  body = ' '.join([body.split('-')[i] for i in range(1, len(body.split('-')))])
  # Tokenise & Remove the final word "reuter"
  body = body.split()[0:-1]
  # Convert to lower case
  body = [str(i).lower() for i in body]
  # Remove punctuation
  body = regex.sub(ur'\p{P}+', '', ' '.join(body)).split()
  # Remove numbers
  body = [i for i in body if not i.isdigit()]
  # Apply spell corrector and transform into TextBlob to apply
  # part of speech tagger
  body = TextBlob(' '.join(body)).correct()
  # Lemmatisation
  lem = WordNetLemmatizer()
  body = [lem.lemmatize(i) for i in body.split()]
  # Remove English stopwords.
  stop = stopwords.words('english')
  body = TextBlob(' '.join([i for i in body if i not in stop]))
  print body.words
  return body


# Run named entity recogniser
def named_entities(tagged_body):
  body = ne_chunk(tagged_body.tags)
  print body
  return body


if __name__ == '__main__':
  main()
