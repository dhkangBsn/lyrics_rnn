import torch
from torch import nn
import numpy as np
import pandas as pd
import MeCab


df = pd.read_csv('./발라드.csv')
text = df['lyrics'].values
wakati = MeCab.Tagger("-Owakati")
text = [wakati.parse(data) for data in text]
print(text[0])


result = []

for sentence in text:
    word_list = sentence.split()
    result += word_list

print(result[:3])
chars = set(result)
print(chars)


# Creating a dictionary that maps integers to the characters
int2char = dict(enumerate(chars))
print(int2char)

# Creating another dictionary that maps characters to integers
char2int = {char: ind for ind, char in int2char.items()}
print(char2int)

# Finding the length of the longest string in our data
maxlen = len(max(text, key=len))
print(maxlen)

for i in range(len(text)):
  while len(text[i]) < maxlen:
      text[i] += ' '
print(text[1])


input_seq = []
for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]

print(input_seq)