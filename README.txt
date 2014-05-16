This repository consists of text mining experimentation carried out for a university data mining assignment. A variety of feature selection methods, classification and clustering may be carried out on the Reuters dataset, as can be found in this repository.

The project is written in Python 2.7. A number of Python modules must be installed before the code can be run. These are listed per file below. All of these can be installed using Python pip.

preprocessing.py:
- BeautifulSoup4
- regex
- nltk
- TextBlob

txtblob.py:
- BeautifulSoup4
- gensim
- TextBlob

scilearn.py:
- numpy
- regex
- BeautifulSoup4
- matplotlib
- sklearn

In addition to these a number of corpora for nltk need to be installed. These are included in the repository and need to be moved to /home/user.

The functions of each of the Python files is described below:
- preprocessing.py: performs preprocessing on the raw data found in data/, including cleaning and lemmatisation. Should be used to produce preprocessing.txt, which is also included.
- txtblob.py: prints out information obtained from bag-of-words, tfidf and topic model feature selection methods. Also displays results of using a naive Bayes and a decision tree classifier from the TextBlob module.
- scilearn.py: performs 10-fold cross validation on 10 different classifiers from the sklearn module. Also performs 4 different clustering algorithms.

All Python files require the path to the data in order to read it in. For example:

python scilearn.py /home/ruth/DataMining/data
