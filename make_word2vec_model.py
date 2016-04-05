import gensim, logging
import codecs
import logging
import sys

class MySentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname):
                yield line.split()

def model_generator(filename):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    lines = MySentences(filename)
    model = gensim.models.Word2Vec(lines, size=200, window=4, min_count=0, workers=4)
    model.save('model.en')

if __name__ == "__main__":
    filename = 'Data/sentences.cleaned.abbreviated'
    model_generator(filename)
