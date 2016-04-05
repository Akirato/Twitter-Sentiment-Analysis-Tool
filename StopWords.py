StopWordsList = []
StopWordsSet = set()

with open("./Data/stopwords.txt") as input_file:
    for input_line_raw in input_file:
        input_tokens = input_line_raw.split(', ')
        StopWordsList.extend(input_tokens)
    StopWordsSet = set(StopWordsList)

def isStopWord(token):
    if token in StopWordsSet:
        return True
    else: 
        return False
