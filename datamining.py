#!/usr/bin/python

import math
import numpy
import regex

from bs4 import BeautifulSoup

from gensim import corpora, models
from gensim.models import ldamodel

from nltk.chunk import ne_chunk
from nltk.corpus import stopwords, wordnet
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.cluster import DBSCAN, KMeans, Ward
from sklearn.cross_validation import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GMM
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn import preprocessing

from textblob import TextBlob, Word
from textblob.classifiers import DecisionTreeClassifier, NaiveBayesClassifier
from textblob.taggers import NLTKTagger

def main():
  # Whether we are using scilearn for features or our own custom.
  scilearn = True
  tfidf = False

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
    if i == 0 or i == 21:
    #if True:
      #addr = '/home/rawr/uni/data_mining/ass/data/reut2-0%s.sgm' % num
      addr = '/home/ruth/uni/dm/ass/data/reut2-0%s.sgm' % num
      new_soup = BeautifulSoup(open(addr))
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
      body = these_bodies[j].text
      cleaned = preprocess(body, scilearn)
      for topic in these_topics[j]:
        topic = str(topic)[3:-4]
        if topic in interested_topics:
          if lewis[j] == 'TRAIN':
            training_set.append([cleaned, topic])
            training_data.append(cleaned)
            training_topics.append(topic)
          elif lewis[j] == 'TEST':
            test_set.append([cleaned, topic])
            test_data.append(cleaned)
            test_topics.append(topic)
        texts.append(cleaned)
        raw_texts.append(body)
        #print cleaned
        topics.append(topic)

  test = 'U.K. MONEY MARKET SHORTAGE FORECAST REVISED DOWN LONDON, March 3 - The Bank of England said it had revised its forecast of the shortage in the money market down to 450 mln stg before taking account of its morning operations. At noon the bank had estimated the shortfall at 500 mln stg. REUTER'
  preprocess(test, scilearn)
  
  topic_dict = {}
  num_topics = len(set(topics))
  for topic in set(topics):
    topic_dict[topic] =  topics.count(topic)
    print topic + ': ' + str(topics.count(topic))
  print 'total number of docs: %d' % len(topics)
  print '***********************'

  # FOR TESTING
  if len(test_data) == 0:
    print 'Making artificial test data'
    text = preprocess('i am a test data, hello!', scilearn)
    topic = 'earn'
    test_set.append([text, topic])
    test_data.append(text)
    test_topics.append(topic)

  if scilearn:
    vect = HashingVectorizer(stop_words='english')
    featured_texts = vect.fit_transform(texts)

    if tfidf:
      vect = TfidfVectorizer(strip_accents='unicode',
                             sublinear_tf=True,
                             stop_words='english')
    else:
      vect = CountVectorizer(strip_accents='unicode',
                             stop_words='english')

    featured_train = vect.fit_transform(training_data)
    print 'Training: n_samples: %d, n_features: %d' % featured_train.shape
    featured_test = vect.transform(test_data)
    print 'Test: n_samples: %d, n_features: %d' % featured_test.shape
    featured_texts = vect.fit_transform(texts)
    print 'Fininshed vectoriser'

    chi = SelectKBest(chi2, 10)
    featured_train = chi.fit_transform(featured_train, training_topics)
    featured_test = chi.transform(featured_test)
    chi = SelectKBest(chi2, 5)
    featured_texts = chi.fit_transform(featured_texts, topics)
    print 'Finished chi^2'

    feature_names = numpy.asarray(vect.get_feature_names())
    print feature_names

    classifiers = [GaussianNB(), MultinomialNB(alpha=0.1),
                   BernoulliNB(), LinearSVC(), RandomForestClassifier(),
                   KNeighborsClassifier(), NearestCentroid()]

    kfolds = KFold(n=len(training_data), n_folds=10, shuffle=True)
    acc_values = {classifier: list() for classifier in classifiers}
    for train, test in kfolds:
      train_text = [featured_train.toarray()[j] for j in train]
      test_text = [featured_train.toarray()[j] for j in test]
      train_topic = [training_topics[j] for j in train]
      test_topic = [training_topics[j] for j in test]
      for classifier in classifiers:
        this_acc = classify(classifier,
                            train_text,
                            train_topic,
                            test_text,
                            test_topic,
                            interested_topics)
        acc_values[classifier].append(this_acc)
    acc_averages = {(i, numpy.mean(acc_values[i])) for i in classifiers}
    acc_stds = {i: numpy.std(acc_values[i]) for i in classifiers}
    
    acc = 0
    best_classifier = None
    for classifier, avg in acc_averages:
      print 'classifier: %s' % classifier.__class__.__name__
      print 'accuracy mean: %f standard deviation: %f' % (avg, acc_stds[classifier])
      if avg > acc:
        acc = avg
        best_classifier = classifier

    classify(best_classifier,
             featured_train,
             training_topics,
             featured_test,
             test_topics,
             interested_topics)

    # **** Clustering ****
    print '****************** Clustering ******************'
    # Use SVD rather than PCA as it is able to work on sparse matrices
    svd = TruncatedSVD(n_components=2)
    dense_texts = svd.fit_transform(featured_texts)
    norm = preprocessing.Normalizer(copy=False)
    dense_texts = norm.fit_transform(dense_texts)
    
    cluster(KMeans(n_clusters=num_topics), featured_texts, topics)
    #cluster(AffinityPropagation(), dense_texts, topics)
    # These methods don't support sparse matrices, so aren't suitable for text mining.
    cluster(DBSCAN(), dense_texts, topics)
    cluster(Ward(n_clusters=num_topics), dense_texts, topics)
    # GMM
    dense_train = svd.fit_transform(featured_train)
    dense_train = norm.fit_transform(dense_train)
    dense_test = svd.fit_transform(featured_test)
    dense_test = norm.fit_transform(dense_test)
    gmm = GMM(n_components=num_topics)
    mappings = {}
    list_topics = list(set(topics))
    for i in range(0, len(list_topics)):
      mappings[list_topics[i]] = i
    train_topics_mapped = []
    for topic in training_topics:
      train_topics_mapped.append(mappings[topic])
    test_topics_mapped = []
    for topic in test_topics:
      test_topics_mapped.append(mappings[topic])
    gmm.means_ = numpy.array([dense_train[
                              train_topics_mapped == i].mean(axis=0)
			   for i in xrange(len(list_topics))])
    gmm.fit(dense_train)
    train_pred = gmm.predict(dense_train)
    test_pred = gmm.predict(dense_test)

    train_acc = numpy.mean(
        train_pred.ravel() == numpy.array(train_topics_mapped).ravel()) * 100
    test_acc = numpy.mean(
        test_pred.ravel() == numpy.array(test_topics_mapped).ravel()) * 100
    print 'train acc = %0.3f' % train_acc
    print 'test acc = %0.3f' % test_acc

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

    DT = DecisionTreeClassifier(training_set)
    print DT.accuracy(test_set)
    print DT.show+informative_features(50)


def classify(classifier,
             training_data,
             training_topics,
             test_data,
             test_topics,
             topics,
             report=True):
  print 'training %s' % str(classifier)
  classifier.fit(training_data, training_topics)
  print 'testing'
  pred = classifier.predict(test_data)
  score = metrics.f1_score(test_topics, pred)
  print 'f1 score: %f' % score
  print 'recall macro: %f' % metrics.recall_score(test_topics, pred, average='macro')
  print 'recall micro: %f' % metrics.recall_score(test_topics, pred, average='micro')
  print 'precision macro: %f' % metrics.precision_score(test_topics, pred, average='macro')
  print 'precision micro: %f' % metrics.precision_score(test_topics, pred, average='micro')
  if report:
    print 'report:'
    print metrics.classification_report(test_topics, pred)
  return metrics.accuracy_score(test_topics, pred)


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


# Preprocessing and cleaning of text bodies
def preprocess(body, scilearn):
  #print body
  # Change to utf-8 encoding
  body = body.encode('utf-8')
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
  if not scilearn:
    cleaned = clean_body(body)
    #tagged = named_entities(cleaned)
    print cleaned
    return clean_body(body)
  else:
    #print ' '.join(body)
    return ' '.join(body)


# Spell correction, lemmatisation and stopwords
def clean_body(body):
  # Remove English stopwords.
  stop = stopwords.words('english')
  body = TextBlob(' '.join([i for i in body if i not in stop]))
  # Apply spell corrector and transform into TextBlob to apply
  # part of speech tagger
  body = TextBlob(' '.join(body.words), pos_tagger=NLTKTagger()).correct()
  # Convert to tags recognisable by the lemmatiser
  body = [(text[0], to_wordnet_tag(text[1])) for text in body.tags]
  body = [(text[0], text[1]) for text in body if text[1] != '']
  # Lemmatisation
  lem = WordNetLemmatizer()
  body = [lem.lemmatize(text[0], text[1]) for text in body]
  return body


# Convert nltk tags to tags recognised by the lemmatizer
def to_wordnet_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


# Run named entity recogniser
def named_entities(tagged_body):
  body = ne_chunk(tagged_body)
  return body


if __name__ == '__main__':
  main()
