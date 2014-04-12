#!/usr/bin/python

from bs4 import BeautifulSoup
from gensim import corpora, models
from gensim.models import ldamodel
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import regex
from textblob import TextBlob, Word

def main():
  soups = []
  raw_texts = []
  texts = []
  topics = []

  interested_topics = {'corn', 'earn', 'acquisitions', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat'}

  #for i in range(0, 22):
  for i in range(0, 1):
    num = str(i)
    if (i < 10): 
      num = "0" + str(i);
    newSoup = BeautifulSoup(open("/home/ruth/uni/dm/ass/data/reut2-0"+num+".sgm"))
    soups.append(newSoup)
    
  for i in range(0, len(soups)):
    thisSoup = soups[i]
    theseBodies = thisSoup.findAll('text')
    theseTopics = thisSoup.findAll('topics')
    print len(theseBodies)
    print len(theseTopics)
    
    #for i in range(1, min(len(theseBodies), len(theseTopics))):
    for i in range(1, 30):
      topic = theseTopics[i].text
      if topic in interested_topics:
        body = theseBodies[i].text
        #print body
        texts.append(clean_body(body))
        raw_texts.append(body)
        #print clean_body(body)
        topics.append(topic)

  print len(texts)
  print len(topics)
  print len(soups)
  bag_of_words(texts)


def bag_of_words(texts):
  # Use dictionary to create 'Bag of words'
  dictionary = corpora.Dictionary([i.split() for i in texts])

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
    uni = text.split()
    vectors.append(dictionary.doc2bow(uni))

  # Apply an LDA Model to the bag of words
  model = ldamodel.LdaModel(vectors, id2word=dictionary, num_topics=10)
  # For example print the probability distribution for the first text
  print model[vectors[0]]
  print model[vectors[1]]
  print model[vectors[2]]


# Method to carry out pre-processing and cleaning of bodies
def clean_body(body):
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
  print body.tags
  return body


# Run named entity recogniser
def named_entities(tagged_body):
  body = ne_chunk(tagged_body.tags)
  print body
  return body


if __name__ == '__main__':
  main()
