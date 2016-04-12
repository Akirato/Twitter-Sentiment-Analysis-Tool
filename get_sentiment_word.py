from easygui import *
import pickle,string,webbrowser
from afinn.afinn import Afinn
from train_and_validate import vectorize


title = "Twitter Sentiment Analysis Tool"

afinn = Afinn(emoticons=True)
afinn2 = Afinn()
def remove_punctuations(s):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return s

def score_afinn(s):
    return afinn.score(s)

def score_bigrams(bigrams):
    score = 0
    if len(bigrams) == 0:
        return score
    for bigram in bigrams:
        if bigram[0].strip() in ['no','not','never','none','nobody','nowhere','nothing','cannot']:
            score+=(-1*afinn.score(bigram[1]))
        else:
            score+=afinn.score(bigram[0]+" "+bigram[1])
    return int(round(float(score)/float(len(bigrams))))

predictor = pickle.load(open('random_for_predictor.pickle','rb'))
inp = enterbox("Enter the sentence to be analyzed:", title)
#inp = raw_input("Enter the sentence to be analysed: ")
while inp!="end":
    if len(inp) > 0:
        inp = remove_punctuations(inp)
        link = 'http://127.0.0.1:8000/sentimentAnalysis/default/show/'
        bigrams = [b for b in zip(inp.split(" ")[:-1], inp.split(" ")[1:])]
        emoticon_score = score_afinn(inp) - afinn2.score(inp)
        train_score = -1 if int(predictor.predict(vectorize(inp))[0]) == 0 else 1
        bigram_score = score_bigrams(bigrams)
        lexical_score = int(score_afinn(inp)/2)
        link = link+inp.replace(" ",'%20')+'/'
        score = train_score + bigram_score + lexical_score + emoticon_score
        if score >= 0 :
            link+='positive/'
        else:
            link+='negative/'

        if train_score == -1:
            link+='negative/'
        else:
            link+='positive/'

        if bigram_score >= 0 :
            link+='positive/'
        else:
            link+='negative/'
 
        if lexical_score < 0:
            link+='negative/'
        elif lexical_score == 0:
            link+='neutral/'
        else:
            link+='positive/'

        if emoticon_score > 0:
            link+='positive'
        elif emoticon_score == 0:
            link+='neutral'
        else:
            link+='negative'

    text = "Positive" if score >= 0 else "Negative"
    choices = ["Continue","Detailed","Quit"]
    reply=buttonbox("Overall Result:"+text,choices=choices)
    if reply=="Quit":
        msgbox("Alright.  Later!", title, ok_button="Quit.")
        quit()
    elif reply == "Detailed":
        webbrowser.open(link,new=2)
#    inp = raw_input("Enter the sentence to be analysed: ")
    inp = enterbox("Enter the sentence to be analyzed:", title)
