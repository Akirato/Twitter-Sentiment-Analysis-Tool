from easygui import *
import pickle
from train_and_validate_bigram import vectorize
predictor = pickle.load(open('bigram_neuralnet_predictor.pickle','rb'))

title = "Twitter Sentiment Analysis Tool"

band = enterbox("Enter the sentence to be analyzed:", title)

#def score(text):
    
#inp = raw_input("Enter the sentence to be analysed: ")
while band!="end":
#    if len(inp) > 0:
#        print predictor.predict(vectorize(inp))
#    msgbox(predictor.predict(vectorize(band)), title, ok_button="Next")
    score_model = list(predictor.predict(vectorize(band)))[0]
#    score_text 
    text = "Positive" if score_model == 1.0 else "Negative"
    if not ccbox(text, title):
        msgbox("Alright.  Later!", title, ok_button="Quit.")
        quit()
    band = enterbox("Enter the sentence to be analyzed:", title)

#    inp = raw_input("Enter the sentence to be analysed: ")
