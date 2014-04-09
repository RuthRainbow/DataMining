#!/usr/bin/python

from bs4 import BeautifulSoup
from gensim import corpora, models
from gensim.models import ldamodel
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
import regex
from textblob import TextBlob, Word

def main():
  soups = []
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
    for i in range(1, 20):
      topic = theseTopics[i].text
      if topic in interested_topics:
        body = theseBodies[i].text
        #print body
        texts.append(clean_body(body))
        #print clean_body(body)
        topics.append(topic)

  print len(texts)
  print len(topics)
  print len(soups)

  dictionary = corpora.Dictionary([[j[0] for j in i] for i in texts])
  # Remove words that only appeared once over all examples
  once_ids = [i for i, j in dictionary.dfs.iteritems() if j == 1]
  #print once_ids
  #dictionary.filter_tokens(once_ids)
  #dictionary.compactify()
  #print(dictionary.token2id)
  # Convert the bodies to sparse vectors
  vectors = []
  for text in texts:
    uni = [i[0] for i in text]
    vectors.append(dictionary.doc2bow(uni))
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
  # Apply spell corrector
  body = TextBlob(' '.join(body)).correct().split()
  # Remove punctuation
  body = regex.sub(ur'\p{P}+', '', ' '.join(body)).split()
  # Remove numbers
  body = [i for i in body if not i.isdigit()]
  # Apply part of speech tagger.
  body = pos_tag(body)
  # Lemmatisation
  lem = WordNetLemmatizer()
  body = [[lem.lemmatize(i[0]), i[1]] for i in body]
  # Remove English stopwords.
  stop = stopwords.words('english')
  body = [i for i in body if i[0] not in stop]
  # Run named entity recogniser
  body = ne_chunk(body, binary=True)
  print body
  return body


if __name__ == '__main__':
  main()
