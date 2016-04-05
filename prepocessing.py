import sys
import csv

def parse_into_sentiments_and_tags(input_file):
    fp = open(input_file, 'rb' )
    sentences = open('sentences','wb')
    tags = open('tags','wb')
    reader = csv.reader( fp, delimiter=',', quotechar='"', escapechar='\\' )
    for row in reader:
        sentences.write(row[5]+'\n')
        tags.write(row[0]+'\n')

def load_into_dict(input_file):
    fp = open(input_file, 'rb' )
    sentences = fp.readlines()
    abbreviations = {}
    for line in sentences:
        if len(line.split(' \t'))>1:
            a,b = line.split(' \t')
        abbreviations[a.strip().lower()] = b.strip()
    return abbreviations

def replace_with_abbreviations(abbreviations,input_file):
    fp = open(input_file,'rb')
    abbreviated = open(input_file+'.abbreviated','wb')
    sentences = fp.readlines()
    for line in sentences:
        m = line.split()
        new_m = []
        for i in m:
            try:
                new_m += [j.lower() for j in abbreviations[(i.strip()).lower()].split()]
            except:
                new_m.append((i.strip()).lower())
        abbreviated.write(' '.join(new_m)+'\n')
    abbreviated.close()
    return None

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "This is not the place for you."
    if sys.argv[1]=='parse' and sys.argv[2]:
        input_file = sys.argv[2]
        parse_into_sentiments_and_tags(input_file)
    if sys.argv[1]=='abbreviations' and sys.argv[2] and sys.argv[3]:
        abbr_file = sys.argv[2]
        input_file = sys.argv[3]
        abbreviations = load_into_dict(abbr_file)
        replace_with_abbreviations(abbreviations,sys.argv[3]) 
