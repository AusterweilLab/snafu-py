import numpy as np
from . import *

def wordFrequency(subj, missing=0.5, data=None):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    # if fluency data are hierarchical, report mean per individual
    if isinstance(subj[0][0], list):
        freqs = []
        excludeds = []
        for l in subj:
            freq, excluded = wordStat(l, missing=missing, data=data)
            freqs.append(np.mean(freq))
            excludeds.append(flatten_list(excluded))
        return freqs, excludeds

    # if fluency data are non-hierarchical, report mean per list
    else:
        freq, excluded = wordStat(subj, missing=missing, data=data)
        return freq, excluded

def ageOfAcquisition(subj, missing=None, data=None):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    # if fluency data are hierarchical, report mean per individual
    if isinstance(subj[0][0], list):
        aoa = []
        excludeds = []
        for l in subj:
            aoa, excluded = wordStat(l, missing=missing, data=data)
            aoas.append(np.mean(aoa))
            excludeds.append(flatten_list(excluded))
        return aoa, excludeds
    # if fluency data are non-hierarchical, report mean per list
    else:
        aoa, excluded = wordStat(subj, missing=missing, data=data)
        return aoa, excluded

def wordStat(subj, missing=None, data=None):
    """One line description here.
    
        Detailed description here. Detailed description here.  Detailed 
        description here.  
    
        Args:
            arg1 (type): Description here.
            arg2 (type): Description here.
        Returns:
            Detailed description here. Detailed description here.  Detailed 
            description here. 
    """
    
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
    
    return word_val, words_excluded
