from . import *

def wordFrequency(subj,missing=None,data=None):
    return wordStat(subj,missing=missing,data=data)

def ageOfAquisition(subj,missing=None,data=None):
    return wordStat(subj,missing=missing,data=data)

def wordStat(subj,missing=None,data=None):
    # load dictionary
    d_val = {}
    with open(data, 'rt', encoding='utf-8-sig') as csvfile:
        # allows comments in file thanks to https://stackoverflow.com/a/14158869/353278
        reader = csv.DictReader(filter(lambda row: row[0]!='#', csvfile), fieldnames=['word','val'])
        for row in reader:
            d_val[row['word']]= float(row['val'])

    word_val = []
    words_excluded = []
    for i in subj: # each list
            temp=[]
            excluded=[]
            for j in i: # each word
                if (j in d_val): # word must be in the list
                    temp.append(d_val[j])
                else: # or their would be excluded
                    if (missing!=None): # case 2: not in the list, substituted by missing
                        temp.append(missing)
                    else:
                        excluded.append(j)
            if(len(temp)>0):
                word_val.append(np.mean(temp))
            words_excluded.append(excluded)
    return np.mean(word_val), words_excluded

