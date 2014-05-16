#!/usr/bin/python

import math
import sys

from bs4 import BeautifulSoup

from gensim import corpora, models
from gensim.models import ldamodel

from textblob.classifiers import DecisionTreeClassifier, NaiveBayesClassifier

interested_topics = {'corn', 'earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat'}


"""
This class applies and prints results for a variety of feature selection methods.
TextBlob classification algorithms are then tested and the results printed.
"""
def main(argv):
  # Saved training and test sets, in the form of (body, topic) tuples
  training_set = []
  test_set = []
  # Used to print topic information and for feature selection
  topics = []
  texts = []

  # Load in the preprocessed data and the raw data as soups
  soups = load_soups(argv[0])
  loaded = load_data()
  done_index = 0

  # Assign all data a topic and lewissplit
  for i in range(0, len(soups)):
    this_soup = soups[i]
    reuters = this_soup.find_all('reuters')
    lewis = []
    these_topics = []
    for reut in reuters:
      lewis.append(reut.get('lewissplit'))
      these_topics.append(reut.topics.findChildren())

    for j in range(1, len(these_topics)):
      cleaned = loaded[done_index]
      done_index += 1
      if len(cleaned) > 0:
        for topic in these_topics[j]:
          # Add the data once per topic, using the lewissplit
          topic = str(topic)[3:-4]
          if topic in interested_topics:
            if lewis[j] == 'TRAIN':
              training_set.append([' '.join(cleaned), topic])
            elif lewis[j] == 'TEST':
              test_set.append([' '.join(cleaned), topic])
            if lewis[j] == 'TRAIN' or lewis[j] == 'TEST':
              topics.append(topic)
              texts.append(cleaned)

  print_topic_info(topics)

  # Feature selection methods:
  bag_of_words(texts)
  tfidf_scores = calc_tf_idf(texts, 10)
  print tfidf_scores[' '.join(texts[0])]
  print tfidf_scores[' '.join(texts[1])]
  print tfidf_scores[' '.join(texts[2])]

  # Classification using TextBlob:
  print 'Naive Bayes:'
  NB = NaiveBayesClassifier(training_set)
  print_classifier_stats(NB, test_set)

  print 'Decision Tree:'
  DT = DecisionTreeClassifier(training_set)
  print_classifier_stats(DT, test_set)


# Load in preprocessed data from file
def load_data():
  loaded = []
  with open('preprocessed.txt', 'r') as f:
    for line in f:
      item = line.split('\', u\'')
      item[0] = item[0][3:]
      item[len(item)-1] = item[len(item)-1][:len(item)-4]
      loaded.append(item)
  print 'loaded %d items' % len(loaded)
  return loaded


# Load in the raw data as soups
def load_soups(base_addr):
  print 'loading soups...'
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


# Calculate the TFIDF value for all given texts
def calc_tf_idf(texts, x):
  print 'Calculating tfidf...'
  all_scores = {}
  for i, text in enumerate(texts):
    scores = {word: tfidf(word, text, texts) for word in text}
    # Sort words by tf-idf 
    sorted_words = sorted(scores.items(), key = lambda x: x[1], reverse = True)
    # Take top x words and store in all_scores
    all_scores[' '.join(text)] = sorted_words[:x]
  return all_scores


# Term frequency
def tf(word, text):
  return float(text.count(word)) / float(len(text))


def num_texts(word, texts):
  return sum(1 for text in texts if word in text)


# Inverse document frequency
def idf(word, texts):
  return math.log(float(len(texts)) / float(1 + num_texts(word, texts)))


def tfidf(word, text, texts):
  return tf(word, text) * idf(word, texts)


# Transform the given texts into a bag of words and apply a topic model
def bag_of_words(texts):
  print 'Creating bag of words...'
  # Use dictionary to create 'Bag of words'
  dictionary = corpora.Dictionary(texts)

  once_ids = [i for i, j in dictionary.dfs.iteritems() if j == 1]
  dictionary.filter_tokens(once_ids)
  # Optionally remove words that appeared in less than 3 documents
  #dictionary.filter_extremes(no_below=3)
  #dictionary.compactify()

  # Convert the bodies to sparse vectors
  vectors = []
  for text in texts:
    vectors.append(dictionary.doc2bow(text))
  print 'Bag of words vector: '
  print vectors[0]

  # Apply an LDA Model to the bag of words
  num_topics = len(interested_topics)
  model = ldamodel.LdaModel(vectors, id2word=dictionary, num_topics=num_topics)
  # For example print the probability distribution for the first 10 texts
  print 'LDA model: '
  for i in xrange(10):
    print model[vectors[i]]
  print 'best 10 topics:'
  print model.show_topics(topics=10, topn=10, formatted=True)


# Print metrics about how a given classifier performs
def print_classifier_stats(classifier, test_set):
  print classifier.accuracy(test_set)
  print classifier.show_informative_features(50)
  print test_set[0]
  print classifier.classify((test_set[0])[0])


if __name__ == '__main__':
  main(sys.argv[1:])
