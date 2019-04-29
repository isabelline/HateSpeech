import argparse
import pandas as pd
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import to_categorical
from ekphrasis.classes.preprocessor import TextPreProcessor 
from ekphrasis.classes.tokenizer import SocialTokenizer 
from ekphrasis.dicts.emoticons import emoticons
import numpy as np
import os
import pickle as pkl



def check_extension(args):
	if args['train'][-3:] == "tsv":
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
			data[key]['label'] = list(df.HS)
			data[key]['y'] = to_categorical(np.asarray(list(df.HS)))
		except:
			print("No Labels in " + key)
		text_all += list(df.text)		
	return text_all


def clean_then_tokenize_text(data):
	text_all = []
	text_processor = TextPreProcessor(
		normalize=['user','url'],)
	for key in data:
		text = data[key]
		a= []
		temp = ""
		for line in text:
			if True:
				line = text_processor.pre_process_doc(line)
				temp=" ".join( text_to_word_sequence(line) )
				a.append(temp)
		data[key]['cln_text'] = a
		text_all +=a
	return text_all


def save_clean_file(args, data):
	for key in data:
		text = data[key]['cln_text']
		with open(os.path.join(args['data_dir'], key+".clean.txt"), 'w') as f:
			for line in text:
				f.write(line+"\n")

def get_text(args, data):
	text_all = read_file(args, data)
	clean_text_all = clean_then_tokenize_text(data)
	save_clean_file(args, data)

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
	print("INFO: save vocab")
	with open(os.path.join(args['data_dir'], "vocab.pkl"), 'wb') as f:
		pkl.dump(word_dict,f)


def prepare_vocab(args, data):
	clean_text_all = get_text(args,data)
	word_dict = get_vocab(clean_text_all)
	add_special_vocab(word_dict)
	data['vocab'] = word_dict
	save_vocab(args, word_dict)


def load_vocab(args):
	print("INFO: load vocab")
	with open(os.path.join(args['data_dir'] ,"vocab.pkl"), 'rb') as f:
		word_dict = pkl.load(f)
	return word_dict


def prepare_X(args, data):
	word_dict = load_vocab(args)
	for key in data:
		x =np.zeros((len(data[key]['text']), int(args['max_len'])))
		text = data[key]['cln_text']
		for i, line in enumerate(text):
			tokens = line.split()
			for j in range(x.shape[1]):
				if i >= len(tokens):
					continue
				else:
					x[i][j] = word_dict[tokens[i]]
		data[key]['x'] = x
		print("INFO: Peek "+key)
		for i in range(3):
			print(data[key]['cln_text'][i])
			print(data[key]['x'][i])
			try:
				print(data[key]['label'][i])
			except:
				print("")

def prepare_Y(args, data):
	print("Y data Already IN!")

def to_numpy(args, data):
	print("INFO: data to numpy file")
	for key in data:
		np.save(os.path.join(args['data_dir'], key+".x"),data[key]['x'])
	try:
		np.save(os.path.join(args['data_dir'], key+".y"), data[key]['y'])
	except:
		print("WARNING: No labels for " + key)

def prepare_glove(args, data):
    glove = dict()
    if args['glove']:
    	print("INFO: Reading Glove Vectors........")
	    with open(os.path.join(args['data_dir'], args['glove']), encoding='utf-8') as f:
		    for line in f:
    		    values = line.split()
        		word = values[0]
       			coefs = np.asarray(values[1:], dtype='float32')
        		glove[word] = coefs
    embedding_weight = np.random.random((len(data['vocab']), coefs.shape[0]))
    cnt = 0
    for word, idx in data['vocab']:
    	if word in glove:
    		embedding_weight[idx] = embeddings_index[word]
    		cnt += 1
    print("INFO: "+str(cnt) +" words in glove out of " +str(len(data['vocab'])) + " total vocabs")




def main():
	parser = argparse.ArgumentParser() 
	parser.add_argument("data_dir")
	parser.add_argument("--train")
	parser.add_argument("--dev")
	parser.add_argument("--test")
	parser.add_argument("max_len")
	parser.add_argument("class_num")
	parser.add_argument("--glove")
	parser.add_argument("class_num")
	parser.add_argument("--max_vocab", default = 30000)
	args = parser.parse_args()
	args = vars(args)
	print("INFO: parameters")
	print(args)

	data = dict()
	if args['train']:
		if True:
			data['train'] = dict()

	if args['dev']:
		if True:
			data['dev'] = dict()

	if args['test']:
		if True:
			data['test'] = dict()

	prepare_vocab(args, data)
	prepare_X(args, data)
	prepare_Y(args, data)
	to_numpy(args, data)
	prepare_glove(args, data)



main()
