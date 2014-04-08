#!/usr/bin/python

from bs4 import BeautifulSoup

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
        texts.append(theseBodies[i].text)
        topics.append(topic)

  print len(texts)
  print len(topics)
  print len(soups)

if __name__ == '__main__':
  main()
