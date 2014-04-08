#!/usr/bin/python

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import regex

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
    
    for i in range(1, min(len(theseBodies), len(theseTopics))):
      topic = theseTopics[i].text
      if topic in interested_topics:
        body = theseBodies[i].text
        texts.append(cleanBody(body))
        print cleanBody(body)
        topics.append(topic)

  print len(texts)
  print len(topics)
  print len(soups)

# Method to carry out pre-processing and cleaning of bodies
def cleanBody(body):
  # Convert to lower case
  body = body.lower()
  # Remove punctuation
  body = regex.sub(ur'\p{P}+', '', body)
  # Remove numbers
  body = ''.join([i for i in body if not i.isdigit()])
  # Remove English stopwords, and remove extra whitespace with join.
  stop = stopwords.words('english')
  body = ' '.join([i for i in body.split() if i not in stop])
  return body

if __name__ == '__main__':
  main()
