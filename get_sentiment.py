import pickle,string
from afinn.afinn import Afinn
from train_and_validate import vectorize

afinn = Afinn(emoticons=True)

def remove_punctuations(s):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return s

def score_afinn(s):
    return afinn.score(s)

predictor = pickle.load(open('random_for_predictor.pickle','rb'))
inp = raw_input("Enter the sentence to be analysed: ")
while inp!="end":
    print(remove_punctuations(inp))
    if len(inp) > 0:
        print predictor.predict(vectorize(inp)),score_afinn(inp)
    inp = raw_input("Enter the sentence to be analysed: ")
