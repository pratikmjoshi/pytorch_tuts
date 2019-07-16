from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script,trace
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import time
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = 'fra.txt'
corpus = os.path.join('./data/fra-eng/',corpus_name)

#Reading samples from the corpus
def printLines(corpus):
	with open(corpus,'r',encoding='iso-8859-1') as f:
		lines = f.readlines()
		for line in lines[:10]:
			print(line)

#printLines(corpus)

#Class to handle vocabulary
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

class Voc:
	def __init__(self,name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {'PAD_TOKEN':0,'SOS_TOKEN':1,'EOS_TOKEN':2}
		self.num_words = 3

	def addSentence(self,sentence):
		for word in sentence.split(' '):
			self.addWord(word)

	def addWord(self,word):
		if word not in self.word2index:
			self.word2index[word] = self.num_words
			self.word2count[word] = 1
			self.index2word[self.num_words] = word
			self.num_words+=1
		else:
			self.word2count[word]+=1


def unicodeToAscii(s):
	return ''.join([c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn'])

def normalizeString(s):
	s = unicodeToAscii(s.lower().strip())
	s = re.sub(r"([.!?])",r" \1",s)
	s = re.sub(r"[^a-zA-Z.!?]",r" ",s)
	return s

#Split data into pairs of French/Hindi and English
def loadData(lang1,lang2,reverse=False):
	print("Reading lines")
	# Read the file and split into lines
	lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
	pairs = [[normalizeString(s) for s in line.split('\t')] for line in lines]
	if reverse:
		pairs = [list(reversed(p)) for p in pairs]
		input_lang = Voc(lang2)
		output_lang = Voc(lang1)
	else:
		input_lang = Voc(lang1)
		output_lang = Voc(lang2)

	return input_lang,output_lang,pairs

#s,t,u = loadData('fra','eng',True)
#print(u[0])

#Training small subset of examples containing the below prefixes

MAX_LENGTH = 10

eng_prefixes = (
	 "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
	return len(p[0].split(' '))<MAX_LENGTH and len(p[1].split(' '))<MAX_LENGTH and p[0].startswith(eng_prefixes)

def filterPairs(pairs):
	
	return [pair for pair in pairs if filterPair(pair)]

#Reading and preprocessing the data
def prepareData(lang1,lang2,reverse=False):
	input_lang,output_lang,pairs = loadData(lang1,lang2)
	print("Read %s sentence pairs" % len(pairs))
	pairs = filterPairs(pairs)
	print("Trimmed to %s pairs" % len(pairs))
	print("Counting words...")
	for pair in pairs:
		input_lang.addSentence(pair[0])
		output_lang.addSentence(pair[1])
	print("Total number of {} words:{:.4f}".format(input_lang.name,input_lang.num_words))
	print("Total number of {} words:{:.4f}".format(output_lang.name,output_lang.num_words))

	return input_lang,output_lang,pairs

input_lang,output_lang,pairs = prepareData('fra','eng',True)
print(random.choice(pairs))

#Neural architecture

#Encoder 
class EncoderRNN(nn.Module):
	def __init__(self,input_size,hidden_size,n_layers=1):
		super(EncoderRNN,self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.embedding = nn.Embedding(input_size,hidden_size)
		self.lstm = nn.LSTM(hidden_size,hidden_size,n_layers)

	def forward(self,input_seq,hidden=None):
		embedded = self.embedding(input_seq)
		output,hidden = self.lstm(embedded,hidden)

		return output,hidden

#Decoder without Attention
class DecoderRNN(nn.Module):
	def __init__(self,hidden_size,output_size,n_layers=1):
		super(DecoderRNN,self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.embedding = nn.Embedding(output_size,hidden_size)
		self.lstm = nn.LSTM(hidden_size,hidden_size,n_layers)
		self.out = nn.Linear(hidden_size,output_size)

	def forward(self,input_seq,hidden=None):
		output = self.embedding(input_seq).view(1,1,-1)
		output = F.relu(output)
		output,hidden = self.lstm(output,hidden)
		output = F.log_softmax(self.out(output[0]),dim=1)
		return output,hidden

#Decoder with Attention
class AttnDecoderRNN(n.Module):
	def __init__(self,hidden_size,output_size,n_layers=1,dropout_p=0.1,max_length=MAX_LENGTH):
		super(AttnDecoderRNN,self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_p = dropout_p
		self.max_length = max_length

		self.embedding = nn.Embedding(self.output_size,self.hidden_size)
		self.attn = nn.Linear(self.hidden_size*2,self.max_length)
		self.attn_combine = nn.Linear(self.hidden_size*2,self.hidden_size)
		self.dropout = nn.Dropout(self.dropout_p)
		self.lstm = nn.LSTM(self.hidden_size,self.hidden_size,n_layers)
		self.out = nn.Linear(self.hidden_size,self.output_size)

	def forward(self,input_seq,encoder_outputs,hidden=None):
		embedded = self.embedding(input_seq)
		embedded = self.dropout(embedded)

		attn_weights = F.softmax(self.attn(torch.cat(embedded[0],hidden[0],1)),dim=1)
		attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

		output = torch.cat(embedded[0],attn_applied[0],1)
		output = self.attn_combine(output).unsqueeze(0)
		output = F.relu(output)

		output,hidden = self.lstm(hidden,output)
		output = F.log_softmax(self.out(output))

		return output,hidden,attn_weights

#Functions to prepare data into tensors
def indexesFromSentences(lang,sentence):
	return [lang.word2index(word) for word in sentence.split(' ')]

def tensorFromSentence(lang,sentence):
	indexes = indexesFromSentences(lang,sentence)
	indexes.append(EOS_TOKEN)
	return torch.tensor(indexes,dtype=torch.long,device=device).view(-1,1)

def tensorsFromPair(input_lang,output_lang,pair):
	input_tensor = tensorFromSentence(input_lang,pair[0])
	output_tensor = tensorFromSentence(output_lang,pair[1])
	return (input_tensor,output_tensor)

#Training process
teacher_forcing_ratio = 0.5

def train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,max_length=MAX_LENGTH):
	encoder_hidden = torch.zeros(1,1,encoder.hidden_size,device=device)
	#Initialize states and variables

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	encoder_outputs = torch.zeros(max_length,encoder.hidden_size,device=device)

	loss = 0
	print_losses = []

	for ei in range(input_length):
		encoder_output,encoder_hidden = encoder(input_tensor[ei],output_tensor[ei],encoder_hidden)
		encoder_outputs[ei] = encoder_output[0,0]

	decoder_input = torch.tensor([[SOS_TOKEN]],device=device)
	decoder_hidden = encoder_hidden

	use_teacher_forcing = True if random.random()<teacher_forcing_ratio else False

	if use_teacher_forcing:
		#Feeding in target as input
		for di in range(target_length):
			decoder_output,decoder_hidden,decoder_attention = decoder(decoder_input,decoder_hidden,encoder_outputs)
			loss+=criterion(decoder_output,target_tensor[di])
			#Teacher forcing
			decoder_input = target_tensor[di]
	else:
		#Without Teacher Forcing, use the previous decoder output as the new decoder input
		for di in range(target_length):
			decoder_output,decoder_hidden,decoder_attention = decoder(decoder_input,decoder_hidden,encoder_outputs)
			_,topi = decoder_output.topk(1)
			decoder_input = topi.squeeze().detach() #Detach from history as input
			
			loss+=criterion(decoder_output,target_tensor[di])
			if decoder_input.item() == EOS_TOKEN:
				break

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item()/target_length

#Helper functions for time
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

#Final iterative training
def trainIters(pairs,encoder,decoder,encoder_optimizer,decoder_optimizer,encoder_n_layers,decoder_n_layers,save_dir,
	n_iterations,batch_size,save_every,loadFileName=None,print_every=100,plot_every=100,learning_rate=0.01,checkpoint=None):
	
	start = time.time()
	plot_losses = []
	print_loss_total = 0 #Reset every print_every
	plot_loss_total = 0 #Reset every plot_every

	encoder_optimizer = optim.SGD(encoder.parameters(),learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(),learning_rate)
	training_pairs = [tensorsFromPair(random.choice(pairs)) for _ in range(n_iterations)]
	criterion = nn.NLLLoss()

	for i in range(1,n_iterations+1):
		training_pair = training_pairs[i+1]
		input_tensor = training_pair[0]
		target_tensor = training_pair[1]

		loss = train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion)
		print_loss_total+=loss
		plot_loss_total+=loss

		if i%print_every == 0:
			print_loss_avg = print_loss_total/print_every
			print_loss_total = 0
			print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iterations),i, i / n_iterations * 100, print_loss_avg))
		if i%plot_every == 0:
			plot_loss_avg = plot_loss_total/plot_every
			plot_loss_total = 0
			print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iterations),i, i / n_iterations * 100, plot_loss_avg))
		if (i%save_every==0):
			directory = os.path.join(save_dir, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
			if not os.path.exists(directory):
				os.makedirs(directory)
			torch.save({
				'iteration': i,
				'en': encoder.state_dict(),
				'de': decoder.state_dict(),
				'en_opt': encoder_optimizer.state_dict(),
				'de_opt': decoder_optimizer.state_dict(),
				'loss': loss
			}, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

	showPlot(plot_losses)

def showPlot(points):
	plt.figure()
	fig, ax = plt.subplots()
	# this locator puts ticks at regular intervals
	loc = ticker.MultipleLocator(base=0.2)
	ax.yaxis.set_major_locator(loc)
	plt.plot(points)
			

#Evaluation 
def evaluate(encoder,decoder,sentence,max_length=MAX_LENGTH):
	with torch.no_grad():
		input_tensor = tensorFromSentence(input_lang,sentence)
		input_length = input_tensor.size()[0]
		encoder_hidden = torch.zeros(1,1,encoder.hidden_size,device=device)

		encoder_outputs = torch.zeros(max_length,encoder.hidden_size,device=device)

		for ei in range(input_length):
			encoder_output,encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
			encoder_outputs[ei] = encoder_output[0,0]

		decoder_input = torch.tensor([[SOS_TOKEN]],dtype=torch.long,device=device)
		decoder_hidden = encoder_hidden

		decoded_words = []
		decoder_attentions = torch.zeros(max_length,max_length)

		for di in range(max_length):
			decoder_output,decoder_hidden,decoder_attention = decoder(decoder_input,decoder_hidden,encoder_outputs)
			decoder_attentions[di] = decoder_attention.data
			_,topi = decoder_output.data.topk(1)
			if topi.item() == EOS_TOKEN:
				decoded_words.append('<EOS>')
				break
			else:
				decoded_words.append(output_lang.index2word[topi.item()])

			decoder_input = topi.squeeze().detach()

		return decoded_words,decoder_attentions[:di+1]































