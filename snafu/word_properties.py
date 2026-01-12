import numpy as np
from . import *

def wordFrequency(subj, missing=0.5, data=None):
    """
    Compute average word frequency for fluency responses.

    This function loads a word frequency dictionary from a file and computes
    average frequency scores for each list or participant. It supports both
    hierarchical and non-hierarchical fluency data.

    Parameters
    ----------
    subj : list
        Fluency data. Can be a list of lists (non-hierarchical) or a list of list of lists (hierarchical).
    missing : float, optional
        Value to substitute for missing words not found in the frequency dictionary (default is 0.5).
    data : str
        Path to a CSV file containing word frequencies. File should have columns: 'word', 'val'.

    Returns
    -------
    list or tuple
        For hierarchical data: a tuple of (list of average frequencies per individual, list of excluded words).
        For non-hierarchical data: a tuple of (list of frequencies per list, list of excluded words).
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
    """
    Compute average age of acquisition (AoA) for fluency responses.

    This function loads a dictionary of age-of-acquisition scores and computes
    average values for each list or participant.

    Parameters
    ----------
    subj : list
        Fluency data. Can be a list of lists (non-hierarchical) or list of list of lists (hierarchical).
    missing : float, optional
        Value to use for words not found in the AoA dictionary. If None, such words are excluded.
    data : str
        Path to a CSV file containing AoA scores. File should have columns: 'word', 'val'.

    Returns
    -------
    list or tuple
        For hierarchical data: a tuple of (list of average AoA per individual, list of excluded words).
        For non-hierarchical data: a tuple of (list of AoA scores per list, list of excluded words).
    """
    # if fluency data are hierarchical, report mean per individual
    if isinstance(subj[0][0], list):
        aoas = []
        excludeds = []
        for l in subj:
            aoa, excluded = wordStat(l, missing=missing, data=data)
            aoas.append(np.mean(aoa))
            excludeds.append(flatten_list(excluded))
        return aoas, excludeds
    # if fluency data are non-hierarchical, report mean per list
    else:
        aoa, excluded = wordStat(subj, missing=missing, data=data)
        return aoa, excluded

def wordStat(subj, missing=None, data=None):
    """
    Compute word-level statistics (e.g., frequency or AoA) from a word-to-value dictionary.

    Loads a dictionary mapping words to numeric values (e.g., frequency, AoA), then computes
    mean values for each list. Handles missing words either by substitution or exclusion.

    Parameters
    ----------
    subj : list of list of str
        List(s) of words for which to compute statistics.
    missing : float, optional
        Value to substitute for missing words. If None, missing words are excluded from computation.
    data : str
        Path to a CSV file with 'word' and 'val' columns.

    Returns
    -------
    tuple
        - word_val : list of float
            Mean value for each list (e.g., frequency or AoA).
        - words_excluded : list of list of str
            Words not found in the dictionary for each list.
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
