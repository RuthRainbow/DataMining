#!/usr/bin/python

import numpy
import regex
import sys

from bs4 import BeautifulSoup

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.cluster import DBSCAN, KMeans, Ward
from sklearn.cross_validation import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GMM
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import preprocessing

interested_topics = {'corn', 'earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat'}


def main(argv):
  """ 
  Which type of vectorisation to use. Possible combinations:
  TFIDF: TFIDF only
  count: Count only (no normalisation)
  count + binary: Binary only (no normalisation)
  """
  tfidf = True
  count = False
  binary = False
  
  # Whether to use lemmatised input
  lemmatise = True

  # All text and topic pairs to be used for clustering
  texts = []
  topics = []
  # Training and test sets based on lewissplit for classification
  training_data = []
  training_topics = []
  test_data = []
  test_topics = []

  soups = load_soups(argv[0])
  loaded = load_data()
  done_index = 0

  # Assign all read in data a topic and lewissplit
  for i in range(0, len(soups)):
    this_soup = soups[i]
    reuters = this_soup.find_all('reuters')
    these_bodies = this_soup.find_all('text')
    lewis = []
    these_topics = []
    for reut in reuters:
      lewis.append(reut.get('lewissplit'))
      these_topics.append(reut.topics.findChildren())
        
    for j in range(1, len(these_topics)):
      body = these_bodies[j]
      if lemmatise:
        cleaned = loaded[done_index]
      else:
        cleaned = preprocess(body)
      done_index += 1
      if len(cleaned) > 0:
        for topic in these_topics[j]:
          topic = str(topic)[3:-4]
          if topic in interested_topics:
            if lewis[j] == 'TRAIN':
              training_data.append(cleaned)
              training_topics.append(topic)
            elif lewis[j] == 'TEST':
              test_data.append(cleaned)
              test_topics.append(topic)
          texts.append(cleaned)
          topics.append(topic)

  print_topic_info(test_topics + training_topics)

  texts = [' '.join(text) for text in texts]
  vect = HashingVectorizer(stop_words='english')
  featured_texts = vect.fit_transform(texts)

  if tfidf and not count:
    vect = TfidfVectorizer(strip_accents='unicode',
                           sublinear_tf=True,
                           ngram_range=(1, 1))
    pipeline = Pipeline([('tfidf', vect)])
  elif count and tfidf:
    vect = CountVectorizer(binary=binary)
    vect2 = TfidfVectorizer(sublinear_tf=True)
    pipeline = Pipeline([('count', vect), ('tfidf', vect2)])
  else:
    vect = CountVectorizer(strip_accents='unicode',
                           binary=binary)
    pipeline = Pipeline([('count', vect)])

  training_data = [' '.join(text) for text in training_data]
  test_data = [' '.join(text) for text in test_data]
  featured_train = pipeline.fit_transform(training_data)
  print 'Training: n_samples: %d, n_features: %d' % featured_train.shape
  featured_test = pipeline.transform(test_data)
  print 'Test: n_samples: %d, n_features: %d' % featured_test.shape
  featured_texts = pipeline.fit_transform(texts)
  print 'Fininshed vectoriser'

  #chi = SelectKBest(chi2, 25)
  #featured_train = chi.fit_transform(featured_train, training_topics)
  #featured_test = chi.transform(featured_test)
  #chi = SelectKBest(chi2, 20)
  #featured_texts = chi.fit_transform(featured_texts, topics)
  print 'Finished chi^2'

  feature_names = numpy.asarray(vect.get_feature_names())
  print feature_names

  classifiers = [GaussianNB(),
                 MultinomialNB(fit_prior=True),
                 #BernoulliNB(binarize=1.0, fit_prior=True),
                 DecisionTreeClassifier(criterion='entropy',
                                        min_samples_split=5,
                                        min_samples_leaf=5),
                 RandomForestClassifier(criterion='entropy',
                                        min_samples_split=5,
                                        min_samples_leaf=5),
                 Perceptron(fit_intercept=True),
                 LinearSVC(fit_intercept=True),
                 KNeighborsClassifier(n_neighbors=10),
                 NearestCentroid()]

  # Perform 10-fold cross validation and save accuracy for each classifier
  kfolds = KFold(n=len(training_data), n_folds=10, shuffle=True)
  acc_values = {classifier: list() for classifier in classifiers}
  featured_train_arr = featured_train.toarray()
  for train, test in kfolds:
    # TODO change away from list constructor so more features can be used (MemoryError)
    train_text = [featured_train_arr[j] for j in train]
    test_text = [featured_train_arr[j] for j in test]
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

  # Discover the best classifier and classify with this using the entire training set
  acc = 0
  best_classifier = None
  for classifier, avg in acc_averages:
    std = acc_stds[classifier]
    print 'classifier: %s' % classifier.__class__.__name__
    print 'accuracy mean: %f standard deviation: %f' % (avg, std)
    left = avg - (1.96 * std)
    right = avg + (1.96 * std)
    print 'confidence: [%f, %f]' % (left, right)
    if avg > acc:
      acc = avg
      best_classifier = classifier
  classify(best_classifier,
           featured_train.toarray(),
           training_topics,
           featured_test.toarray(),
           test_topics,
           interested_topics)
 
  print '****************** Clustering ******************'
  # Use SVD to reduce to sparse vectors to dense vectors and reduce the number of features
  svd = TruncatedSVD(n_components=3)
  dense_texts = svd.fit_transform(featured_texts)
  norm = preprocessing.Normalizer(copy=False)
  dense_texts = norm.fit_transform(dense_texts)
  num_topics = len(set(topics))

  cluster(KMeans(n_clusters=num_topics), featured_texts, topics)
  # These methods don't support sparse matrices, so aren't suitable for text mining.
  cluster(DBSCAN(), dense_texts, topics)
  cluster(Ward(n_clusters=num_topics), dense_texts, topics)
  
  # GMM
  list_topics = list(set(topics))
  gmm(featured_train, featured_test, training_topics, test_topics, svd, norm, num_topics, list_topics)


# Preprocessing and cleaning of text bodies
def preprocess(body):
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
  return body


def load_data():
  loaded = []
  with open('preprocessed.txt', 'r') as f:
    for line in f:
      item = line.split('\', u\'')
      item[0] = item[0][3:]
      item[len(item)-1] = item[len(item)-1][:len(item)-4]
      loaded.append(item)
  return loaded


def load_soups(base_addr):
  soups = []
  for i in range(0, 22):
    num = str(i)
    if (i < 10): 
      num = "0" + str(i);
    if True:
      addr = base_addr + '/reut2-0%s.sgm' % num
      new_soup = BeautifulSoup(open(addr))
      soups.append(new_soup)
  print 'loaded %d soups' % len(soups)
  return soups


def print_topic_info(topics):
  topic_dict = {}
  num_topics = len(set(topics))
  for topic in set(topics):
    topic_dict[topic] =  topics.count(topic)
    print topic + ': ' + str(topics.count(topic))
  print 'total number of docs: %d' % len(topics)
  print '***********************'


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
  acc = metrics.accuracy_score(test_topics, pred)
  print 'accuracy: %f' % acc
  return acc


def cluster(classifier, data, topics, make_silhouette=False):
  print str(classifier)
  clusters = classifier.fit_predict(data)
  labels = classifier.labels_
  print 'Homogeneity: %0.3f' % metrics.homogeneity_score(topics, labels)
  print 'Completeness: %0.3f' % metrics.completeness_score(topics, labels)
  print 'V-measure: %0.3f' % metrics.v_measure_score(topics, labels)
  print 'Adjusted Rand index: %0.3f' % metrics.adjusted_rand_score(topics, labels)
  print 'Silhouette test: %0.3f' % metrics.silhouette_score(data, labels)
  print ' ***************** '
  
  silhouettes = metrics.silhouette_samples(data, labels)
  num_clusters = len(set(clusters))
  print 'num clusters: %d' % num_clusters
  print 'num fitted: %d' % len(clusters)

  if make_silhouette:
    order = numpy.lexsort((-silhouettes, clusters)) 
    indices = [numpy.flatnonzero(clusters[order] == num_clusters) for k in range(num_clusters)]
    ytick = [(numpy.max(ind)+numpy.min(ind))/2 for ind in indices]
    ytickLabels = ["%d" % x for x in range(num_clusters)]
    cmap = cm.jet( numpy.linspace(0,1,num_clusters) ).tolist()
    clr = [cmap[i] for i in clusters[order]]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.barh(range(data.shape[0]), silhouettes[order], height=1.0,   
            edgecolor='none', color=clr)
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.yticks(ytick, ytickLabels)
    plt.xlabel('Silhouette Value')
    plt.ylabel('Cluster')
    plt.savefig('cluster.png')


def gmm(featured_train, featured_test, training_topics, test_topics, svd, norm, num_topics, list_topics):
  print 'GMM'
  dense_train = svd.fit_transform(featured_train)
  dense_train = norm.fit_transform(dense_train)
  dense_test = svd.fit_transform(featured_test)
  dense_test = norm.fit_transform(dense_test)
  gmm = GMM(n_components=num_topics)
  train_topics_mapped = map_topics_to_nums(training_topics)
  test_topics_mapped = map_topics_to_nums(test_topics)
  gmm.fit(dense_train)
  train_pred = gmm.predict(dense_train)
  test_pred = gmm.predict(dense_test)
  train_acc = numpy.mean(
      train_pred.ravel() == numpy.array(train_topics_mapped).ravel()) * 100
  test_acc = numpy.mean(
      test_pred.ravel() == numpy.array(test_topics_mapped).ravel()) * 100
  print 'training accuracy = %0.3f' % train_acc
  print 'test accuracy = %0.3f' % test_acc


def map_topics_to_nums(topics_to_map):
  mappings = {}
  list_topics = list(interested_topics)
  for i in range(0, len(list_topics)):
    mappings[list_topics[i]] = i
  mapped_topics = []
  for topic in topics_to_map:
    mapped_topics.append(mappings[topic])
  return mapped_topics


if __name__ == '__main__':
  main(sys.argv[1:])
