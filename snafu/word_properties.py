import csv
import numpy as np

def wordSetup(root_path):
	global d_freq
	global d_aoa
	d_freq = {}
	d_aoa = {}
	with open(root_path+"/words/words.csv", 'rb') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if(row['freq']=='#N/A'):
				d_freq[row['word']]= -1
			else:
				d_freq[row['word']]= float(row['freq'])

			if(row['aoa_mean']=='NA'):
				d_aoa[row['word']]= -1
			else:
				d_aoa[row['word']]= float(row['aoa_mean'])


def getWordFreq(subj):
	word_freq = []
	for i in subj: # each list
	    for j in i: # each word
	        temp=[]
	        if ( (j in d_freq) and (d_freq[j]!=-1) ):
	            temp.append(d_freq[j])
	    if(len(temp)>0):
	    	word_freq.append(np.mean(temp))
	return np.mean(word_freq)

def getWordAoa(subj):
	word_aoa = []
	for i in subj: # each list
	    for j in i: # each word
	        temp=[]
	        if ( (j in d_aoa) and (d_aoa[j]!=-1) ):
	            temp.append(d_aoa[j])
	    if(len(temp)>0):
	    	word_aoa.append(np.mean(temp))
	return np.mean(word_aoa)