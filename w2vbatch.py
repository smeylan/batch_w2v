#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gensim
import os
import itertools
import pandas
import time
import argparse
import json

def expandgrid(*itrs):
	''' Return the cross poduct of all list arguments ''' 
	product = list(itertools.product(*itrs))
	return product

def trainModel(paramSet, sentences):
	start_time = time.time()
	index = paramSet['index']
	paramSet.pop("index", None)		
	filename = '_'.join([x + '-' + str(paramSet[x]) for x in colnames])
	print ('Training W2V model' + str(index)+ ' of ' +str(numModels)+': '+filename)	
	sg_model = gensim.models.Word2Vec(
		sentences = sentences, 
		size = paramSet['size'], 
		window = paramSet['window'], 
		min_count = paramSet['mc'], 
		workers = paramSet['workers'], 
		sg = paramSet['sg'], 
		hs = paramSet['hs'])	
	print('Saving model...')		
	sg_model.save(os.path.join(paramSet['outputPath'], filename))
	print('Elapsed: '+str((time.time() - start_time) / 3600) + ' hours')


if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--ctrl", type=str, help="name of .ctrl json")
	args = argparser.parse_args()

	print('Loading control .json')
	with open(args.ctrl) as ctrl_json:
		ctrl = json.load(ctrl_json)

	'''.ctrl file must have:
	inputCorpus: plaintext on which the model is trained
	outputPath: path to put the trained models
	workers: number of threads used to train the model

	A full cross of the following parameters will be run: 

	size: dim of feature vectors
	window: scope of the context (n words on either side)
	mc: filter vocab to words with n or more appearances
		sg: #1 is skipgram, 0 is CBOW
		neg: number of negative samples. 0 = hierarchical softmax

	'''
	if not os.path.exists(ctrl['inputCorpus']):
		raise ValueError('Cannot find the input corpus')

	if not os.path.exists(ctrl['outputPath']):
		os.makedirs(ctrl['outputPath'])


	print('Building the parameter space...')
	fullcross = expandgrid(ctrl['parameters']['size'], ctrl['parameters']['window'], ctrl['parameters']['mc'], ctrl['parameters']['sg'], ctrl['parameters']['neg'])
	colnames = ['size', 'window', 'mc', 'sg', 'neg']
	paramTable = pandas.DataFrame(fullcross, columns = colnames)
	paramTable['workers'] = ctrl['parameters']['workers']
	paramTable['hs'] = [0 if x > 0 else 1 for x in paramTable['neg']] #hs = 1 when neg is 0    
	paramTable['outputPath'] = ctrl['outputPath']
	numModels = paramTable.shape[0]
	paramTable['index'] = range(numModels)
	colnames = colnames + ['hs'] # fields to use for building the filenames
	params = []
	for row in paramTable.iterrows():
		params.append(row[1].to_dict())
	print('Training '+str(len(params))+' models')

	print('Loading '+ ctrl['inputCorpus'] + '...')
	sentences = gensim.models.word2vec.LineSentence(ctrl['inputCorpus'])    

	# call to paramSet is not parallelized because the Gensim training is parallelized (with n workers)
	[trainModel(paramSet, sentences) for paramSet in params]