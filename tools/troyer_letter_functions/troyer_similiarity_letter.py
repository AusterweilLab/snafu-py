import pickle
import pandas as pd
import numpy as np
import nltk


### Initialization ###
word_list = [line.rstrip('\n').lower() for line in open('letter_fluency_words.txt')]        # Set of all spoken VF words            
word_list = sorted(word_list)

entries = nltk.corpus.cmudict.entries()                                                     # Create pronunciation dictionary from NLTK
s2p = dict((k,v) for k,v in entries if k in word_list)                            

with open("letter_f_pronunciations.txt", "r") as fh:                                        # Add pronunciations from local text file                      
    lines = [ el.split('  ') for el in fh.readlines() ]
    tups = [ (el[0].strip(), [ x for x in el[1].strip().split(' ') ]) for el in lines ]
    f_pronuns = dict(tups)
f_pronuns =  {k.lower(): v for k, v in f_pronuns.items()}
s2p = {**s2p,**f_pronuns}


### Functions ###
def strip_emphasis(p):
    return ''.join([ c for c in p if c not in '0123456789' ]).strip()

def memoize(function):
    cache = {}
    def decorated_function(*args):
        if args in cache:
            return cache[args]
        else:
            val = function(*args)
            cache[args] = val
            return val
    return decorated_function

@memoize
def rhyme(inp, level, pronunciations=s2p):
    syllables = [(word, syl) for word, syl in pronunciations.items() if word == inp]
    rhymes = []
    for (word, syllable) in syllables:
        rhymes += [word for word, pron in pronunciations.items() if pron[-level:] == syllable[-level:]]
    return set(rhymes)

def doTheyRhyme(word1, word2, level=2):                                                     # Level was set to 2 (1-5 scale, where a level of 5 only creates links essentially between homonyms)
    return word1 in rhyme(word2, level)                                                     # https://stackoverflow.com/questions/25714531/find-rhyme-using-nltk-in-python

def homonyms(w1,w2,pronunciations=s2p):
    if pronunciations[w1] == pronunciations[w2]:
        return True
    else: return False

def same_initial_letters(w1,w2):
    if w1[0:2] == w2[0:2]:
        return True
    else: return False

def one_vowel_difference(w1,w2,pronunciations=s2p):
    p1 = pronunciations[w1]
    p2 = pronunciations[w2]
    diffs = [ sound for sound in p1 + p2 if (sound in p1 and not sound in p2) or (sound in p2 and not sound in p1) ]
    if len(diffs) == 1 and diffs[0][0] in 'AEIOU':
        return True
    else: return False

def islinked(w1,w2):
    if same_initial_letters(w1,w2) or homonyms(w1,w2) or doTheyRhyme(w1,w2) or one_vowel_difference(w1,w2):
        return 1
    else:
        return 0


### Main ###
lst_length = len(word_list)
mat = np.zeros(shape=(lst_length, lst_length))

for i in range(lst_length): #row
    print(i,'/',lst_length-1)
    for j in range(lst_length): #column
        if i == j:
            continue
        else:

            w1 = word_list[i]
            w2 = word_list[j]

            mat[i,j] = islinked(w1,w2)

adj_mat = pd.DataFrame(mat, index = word_list, columns = word_list)
adj_mat.to_csv('/Users/jbushnel/Documents/troyer/troyer_letter_adj_mat.csv')

print(adj_mat)
