import argparse
import pandas as pd
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from ekphrasis.classes.preprocessor import TextPreProcessor 
from ekphrasis.classes.tokenizer import SocialTokenizer 
from ekphrasis.dicts.emoticons import emoticons
import numpy as np
import os
import pickle as pkl



def check_extension(args):
	if args.train[-3:] == "tsv":
		return "\t"
	else:
		return ","

def read_file(args, data):
	sep = check_extension(args)
	text_all = []
	for key in data:
		file = os.path.join(args['data_dir'], args[key])
		df = pd.read_csv(file, sep=sep, encoding="utf-8")
		data[key]['text'] = list(df.text)
		try:
			data[key]['label'] = list(df.label)
			data[key]['y'] = to_categorical(np.asarray(list(df.label)))
		except:
			print("No Labels in " + key)
		text_all += df.text		
	return text_all


def clean_then_tokenize_text(data):
	text_all = []
	for key in data:
		text = data[key]
		a= []
		for line in text
			if true
				temp  =" ".join( text_to_word_sequence(line) ) 
				a.append(temp)
		data[key]['cln_text'] = a
		text_all +=a
	return text_all


def save_clean_file(data):
	for key in data:
	text = data[key]['cln_text']
	with open(os.path.join(args['data_dir'], key+".clean.txt"), 'w') as f:
		for line in text:
			f.write(line+"\n")

def get_text(args, data):
	text_all = read_file(args, data)
	clean_text_all = clean_then_tokenize_text(data)
	save_clean_file(data)

	return clean_text_all

def get_vocab(text_all):
	tokenizer = Tokenizer()     
	tokenizer.fit_on_texts(text_all)
	word_idx = tokenizer.word_index
	return word_idx


def add_special_vocab(word_dict):
	word_dict['<PAD>'] = 0
	word_dict["<UNK>"] = len(word_dict)


def save_vocab(args, word_dict):
	print("----------------save vocab----------------")
	with open(os.path.join(args.data_dir, "vocab.txt"), 'w') as f:
		pkl.dump(word_dict,f)


def prepare_vocab(args, data):
	clean_text_all = get_text(args)
	word_dict = get_vocab(clean_text_all)
	add_special_vocab(word_dict)	
	save_vocab(args, word_dict)


def load_vocab(args):
	print("---------------load vocab--------------")
	with open(args.data_dir + "vocab.txt", 'r') as f:
		word_dict = pkl.load(f)
	return word_dict


def prepare_X(args, data):
	word_dict = load_vocab(args)
	for key in data:
		x =  np.zeros((len(data[key]['text'], args['max_len'])))
		text = data[key]['cln_text']
		for i, line in x.shape[0]:
			tokens = line.split()
			for j in x.shape[1]:
				if i >= len(token):
					continue
				else:
					x[i][j] = word_dict[token[i]]
		data[key]['x'] = x

def prepare_Y():
	print("Already Done!")

def to_numpy(data):
	print("----------data to numpy-----------")
	for key in data:
		np.save(os.path.join(args['data_dir'], key+".x"),data[key]['x'])
	try:
		np.save(os.path.join(args['data_dir'], key+".y"), data[key]['y'])
	except: 
		print("No labels for " + key)

def main():
	parser = argparse.ArgumentParser() 
	parser.add_argument("data_dir")
	parser.add_argument("--train")
	parser.add_argument("--dev")
	parser.add_argument("--test")
	parser.add_argument("max_len")
	parser.add_argument("class_num")
	parser.add_argument("--max_vocab", default = 30000)
	args = parser.parse_args()

	data = dict()
	if train:
		if true:
			data['train'] = dict()

	if dev: 
		if true:
			data['dev'] = dict()

	if test: 
		if true:
			data['test'] = dict()

	prepare_vocab(args, data)
	prepare_X()
	prepare_Y()
	to_numpy(data)



main()