import csv
import numpy as np

def wordSetup(freqfile,aoafile,freq_sub): #freqfile and aoafile are paths to the csv files containing word frequency and aoa dictionary
	global d_freq
	global d_aoa
	d_freq = {}
	d_aoa = {}
	with open(freqfile, 'rb') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
                    d_freq[row['word']]= float(row['freq'])

	with open(aoafile, 'rb') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
                    d_aoa[row['word']]= float(row['aoa_mean'])


def getWordFreq(subj,freq_sub):
	word_freq = []
	words_excluded = []
	for i in subj: # each list
		temp=[]
		excluded=[]
		for j in i: # each word
			if (j in d_freq): # case 1: in the list
                            temp.append(d_freq[j])
			else: 
                            if(j not in d_freq and freq_sub!=-1): # case 2: not in the list, substituted by freq_sub
                                    temp.append(freq_sub)
                            else: # case 3: excluded
                                    excluded.append(j)
		if(len(temp)>0):
			word_freq.append(np.mean(temp))
		words_excluded.append(excluded)
	return np.mean(word_freq), words_excluded

def getWordAoa(subj):
	word_aoa = []
	words_excluded = []
	for i in subj: # each list
		temp=[]
		excluded=[]
		for j in i: # each word
                    if (j in d_aoa): # word must be in the list
                        temp.append(d_aoa[j])
                    else: # or their would be excluded
                        excluded.append(j)
		if(len(temp)>0):
                    word_aoa.append(np.mean(temp))
		words_excluded.append(excluded)
	return np.mean(word_aoa), words_excluded
