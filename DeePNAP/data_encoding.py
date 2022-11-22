# Encoding 
# filtered Data
# Protein Sequence Maxlength = 1000, Nucleic Acid Sequence Maxlength = 75, 
# protein seq consist only 20 standard aa and Nucleic acid seq consist only ATCGU
# required imports
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Bidirectional, LSTM, Reshape
from sklearn.model_selection import train_test_split
from tensorflow.keras import activations
from sklearn.utils import shuffle
from keras import backend as K
from keras.utils import np_utils
import pickle

def mutate_protein(wild, mut):
  muts = mut.split(",")
  additions = []
  ordo = 1000

  for mutation in muts:
    mutation = mutation.strip() #There might be some spaces after doing the split
    if mutation=="": #Skips the iteration if the mutation is empty
      continue

    if "del" == mutation[:3]: #Deletion mutation

      if "-" in mutation: #Range Deletion. Eg, delN5-N10
        mutation = mutation[3:].split("-") #This will give the two parts, say N5 and N10 seperately
        mutation = [(int(i[1:]) - 1) for i in mutation] #Getting the 5 and 10 from N5 and N10. -1 cuz Computer counts from 0
        wild = wild[:mutation[0]] + "O"*(mutation[1]- mutation[0]) + wild[mutation[1]:] #Replacing the characters at index 4,5,6,7,8,9 with an X

      else: #Point deletion
        del_pos = int(mutation[4:]) - 1 #the first three characters are del. 4th is the Amino acid. Rest all are the position number.
        wild = wild[:del_pos] + "O" + wild[(del_pos+1):] #Replaces that single point with an X

    else: #Assumed to be a replacement if there's no del.

      num = ""
      replace = ""
      for j in mutation[1:]: #i[1:] starts iterating from the character after the amino acid. Contains only the position and Mutation
        if j.isdigit():
          num += j
        else:
          replace += j #This will take up the value of the thing that needs to be replaced
      num = int(num)
      wild = wild[:num] + chr(ordo) + wild[(num+1):] #chr(odro) should give us a random character. We put a random character there for now.
      additions.append(replace) #We store the character, and what needs to go there.
      ordo += 1 #Update odro, so that the next mutation will have a new character

  #The wild string has been marked with deletions and replacements. We just have to replace them one by one now.

  #Replacing the Random character with the specified protein.
  ordo = 1000
  for i in range(len(additions)):
    wild = wild.replace(chr(i+ordo), additions[i])
  #Replacing the points marked with X with an empty string
  wild = wild.replace("O", "")

  #Returning the updated protein after doing all the work.
  return wild




# Protein Encoder
def prot_encoding(protein_sequence):
  protein_sequence = protein_sequence.replace(" ", "")
  protein_sequence = protein_sequence.replace(",", "")
  protein_sequence = protein_sequence.replace("\n", "")
  protein_sequence = protein_sequence.replace("", "")
  protein_sequence = protein_sequence.upper()
  amino_acids=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S','T', 'V', 'W', 'Y']
  nlength=len(protein_sequence)
  maxseq=1000
 # generates a array of dimension 1000x22, with value 0
  inputs=np.zeros([maxseq,20])
  ii=0
  for i in protein_sequence:
  # gives the index of the amino acid in the sequence
    j=amino_acids.index(i)
  # takes in the position of amino acid in sequence and makes embedding matrix from it
    #k=p_encoded[j,:]
    k=np_utils.to_categorical(j,num_classes=20)
    inputs[ii,:]= k
    ii += 1
  return inputs

# NA Encoder
def dr_encoding(dr_sequence):
  dr_sequence = dr_sequence.replace(" ", "")
  dr_sequence = dr_sequence.replace(",", "")
  dr_sequence = dr_sequence.replace("\n", "")
  dr_sequence = dr_sequence.replace("", "")
  nucleo_base=['A', 'T', 'C', 'G', 'U']
  nlength=len(dr_sequence)
  maxseq=75
  dr_sequence = dr_sequence.upper()
 # generates a array of dimension 1000x22, with value 0
  inputs=np.zeros([maxseq,5])
  ii=0
  for i in dr_sequence:
  # gives the index of the amino acid in the sequence
    j=nucleo_base.index(i)
  # takes in the position of amino acid in sequence and makes embedding matrix from it
  # k=drencoded[j,:]
    k=np_utils.to_categorical(j,num_classes=5)
    inputs[ii,:]= k
    ii += 1
  return inputs
