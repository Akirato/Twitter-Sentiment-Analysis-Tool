import gensim,numpy
import codecs
import sys
import logging

def vectorize(lang_tag,filename):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec.load('model.'+lang_tag)
    print "The model is loaded."
    data_file = codecs.open(filename,'rb',encoding='utf-8')
    data = data_file.read()
    print "Reading the data file is done."

    word_vector_file = codecs.open(filename+'.word.vectors.'+lang_tag,'wb',encoding='utf-8')
    done,count = [],0
    unsolved_default_vector = numpy.array([0 for i in range(0,200)])

    print "Starting the vectorization."
    for word in set(data.split()):
        try:
            word_vector_file.write(word+'\n'+str(model[word])+'\n')
        except:
            count+=1
            word_vector_file.write(word+'\n'+str(unsolved_default_vector)+'\n')

    print count
    word_vector_file.close()

if __name__ == '__main__':
    script,lang_tag,filename = sys.argv
    vectorize(lang_tag,filename)
