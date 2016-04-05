import pickle
from train_and_validate import vectorize

predictor = pickle.load(open('random_for_predictor.pickle','rb'))
inp = raw_input("Enter the sentence to be analysed: ")
while inp!="end":
    if len(inp) > 0:
        print predictor.predict(vectorize(inp))
    inp = raw_input("Enter the sentence to be analysed: ")
