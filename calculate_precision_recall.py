import pickle,string,csv,numpy
from afinn.afinn import Afinn
from train_and_validate import vectorize
from sklearn.metrics import recall_score,precision_score

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
fp = open('Data/sentiment.csv', 'rb' )
reader = csv.reader( fp, delimiter=',', quotechar='"', escapechar='\\' )
count = 0
precision,recall = 0,0
for row in reader:
    actual = 1 if int(row[0])>0 else 0
    inp = row[5]
    if len(inp) > 0:
        inp = remove_punctuations(inp)
        bigrams = [b for b in zip(inp.split(" ")[:-1], inp.split(" ")[1:])]
        emoticon_score = score_afinn(inp) - afinn2.score(inp)
        train_score = -1 if int(predictor.predict(vectorize(inp))[0]) == 0 else 1
        bigram_score = score_bigrams(bigrams)
        lexical_score = int(score_afinn(inp)/2)
        score = train_score + bigram_score + lexical_score + emoticon_score
        given = 1 if score>=0 else 0
	if given == actual:
            precision+=1
        count+=1
        if count%10000==0:
            print count

print "Precision and Recall: ",float(precision)/float(count)

  
