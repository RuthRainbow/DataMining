#!/usr/bin/python

import regex
import sys

from bs4 import BeautifulSoup

from nltk.chunk import ne_chunk
from nltk.corpus import stopwords, wordnet
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

from textblob import TextBlob, Word
from textblob.taggers import NLTKTagger

interested_topics = {'corn', 'earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat'}


def main(argv):
  soups = load_soups(argv[0])

  # Print each cleaned body which we are interested in
  for i in range(0, len(soups)):
    this_soup = soups[i]
    these_bodies = this_soup.find_all('text')
    reuters = this_soup.find_all('reuters')
    these_topics = []
    for reut in reuters:
      these_topics.append(reut.topics.findChildren())
    for j in range(1, len(these_topics)):
      body = these_bodies[j].text
      # Ignore entries with empty bodies
      if body:
        cleaned = preprocess(body)
        for topic in these_topics[j]:
          topic = str(topic)[3:-4]
          if topic in interested_topics:
            print cleaned


def load_soups(base_addr):
  for i in range(0, 22):
    num = str(i)
    if (i < 10): 
      num = "0" + str(i);
    if True:
      addr = base_addr + '/reut2-0%s.sgm' % num
      new_soup = BeautifulSoup(open(addr))
      soups.append(new_soup)


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
  return lemmatise(body)


# Spell correction, lemmatisation and stopwords
def lemmatise(body):
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
  main(sys.argv[1:])