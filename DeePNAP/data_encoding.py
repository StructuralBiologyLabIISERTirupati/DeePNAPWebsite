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

from tensorflow.keras import activations

from keras import backend as K
import keras.utils as np_utils

drencoded=np.array([[1, 1, 0, 0],
[1, 0, 1, 0],
[1, 0, 0, 1],
[0, 1, 1, 0],
[0, 1, 0, 1],
[0, 0, 1, 1]])

p_encoded=np.array([
 [1, 1, 1, 0, 0, 0, 0, 0, 1], #A
 [1, 1, 0, 1, 0, 0, 0, 1, 0], #C
 [1, 1, 0, 0, 1, 0, 1, 0, 0], #D
 [1, 1, 0, 0, 0, 1, 1, 0, 0], #E
 [1, 0, 1, 1, 0, 0, 0, 0, 1], #F
 [1, 0, 1, 0, 1, 0, 0, 0, 1], #G
 [1, 0, 1, 0, 0, 1, 1, 0, 0], #H
 [1, 0, 0, 1, 1, 0, 0, 0, 1], #I
 [1, 0, 0, 1, 0, 1, 1, 0, 0], #K
 [1, 0, 0, 0, 1, 1, 0, 0, 1], #L
 [0, 1, 1, 1, 0, 0, 0, 0, 1], #M
 [0, 1, 1, 0, 1, 0, 0, 1, 0], #N
 [0, 1, 1, 0, 0, 1, 0, 0, 1], #P
 [0, 1, 0, 1, 1, 0, 0, 1, 0], #Q
 [0, 1, 0, 1, 0, 1, 1, 0, 0], #R
 [0, 1, 0, 0, 1, 1, 0, 1, 0], #S
 [0, 0, 1, 1, 1, 0, 0, 1, 0], #T
 [0, 0, 1, 1, 0, 1, 0, 0, 1], #V
 [0, 0, 1, 0, 1, 1, 0, 1, 0], #W
 [0, 0, 0, 1, 1, 1, 0, 1, 0]]) #Y





def mutate_protein(wild, mut):
    """
    Mutates a protein sequence based on given mutations.

    Args:
        wild (str): The wild-type protein sequence.
        mut (str): A string containing mutations separated by commas.
                   Each mutation can be a deletion or a replacement.

    Returns:
        str: The mutated protein sequence.

    """
    muts = mut.split(",")  # Splitting the mutations string into individual mutations
    additions = []  # List to store replacement characters for each mutation
    ordo = 1000  # Starting value for generating random characters for replacements

    for mutation in muts:
        mutation = mutation.strip()  # Removing any leading or trailing spaces from the mutation

        if mutation == "":  # Skips the iteration if the mutation is empty
            continue

        if "del" == mutation[:3]:  # Deletion mutation

            if "-" in mutation:  # Range Deletion. Eg, delN5-N10
                mutation = mutation[3:].split("-")  # Splitting the range into two parts
                mutation = [(int(i[1:]) - 1) for i in mutation]  # Extracting the start and end positions
                wild = wild[:mutation[0]] + "O" * (mutation[1] - mutation[0]) + wild[mutation[1]:]
                # Replacing the characters between the start and end positions with "O"

            else:  # Point deletion
                del_pos = int(mutation[4:]) - 1  # Extracting the position to be deleted
                wild = wild[:del_pos] + "O" + wild[(del_pos + 1):]  # Replacing the character at that position with "O"

        else:  # Assumed to be a replacement if there's no "del"
            num = ""
            replace = ""
            for j in mutation[1:]:
                if j.isdigit():
                    num += j
                else:
                    replace += j
            num = int(num)  # Extracting the position to be replaced
            wild = wild[:num] + chr(ordo) + wild[(num + 1):]
            # Replacing the character at that position with a random character represented by chr(ordo)
            additions.append(replace)  # Storing the character to be replaced

            ordo += 1  # Updating ordo for generating the next random character

    # The wild string has been marked with deletions and replacements.
    # We just have to replace them one by one now.

    # Replacing the random characters with the specified proteins.
    ordo = 1000
    for i in range(len(additions)):
        wild = wild.replace(chr(i + ordo), additions[i])

    # Removing the "O" characters representing deletions
    wild = wild.replace("O", "")

    # Returning the updated protein after performing all the mutations
    return wild


def dr_encoding(dr_sequence):
    """
    Encodes a DNA/RNA sequence into a numerical representation using one-hot encoding.

    Args:
        dr_sequence (str): DNA/RNA sequence to be encoded.

    Returns:
        numpy.ndarray: Numerical representation of the input sequence using one-hot encoding.

    """

    nucleo_base = ['A', 'T', 'C', 'G', 'U']
    nlength = len(dr_sequence)
    maxseq = 75

    # Generate an array of dimension 75x5, initialized with zeros
    inputs = np.zeros([maxseq, 5])

    ii = 0
    for i in dr_sequence:
        # Get the index of the nucleotide base in the sequence
        j = nucleo_base.index(i)

        # Convert the index into a one-hot encoded matrix
        k = np_utils.to_categorical(j, num_classes=5)

        # Store the one-hot encoded matrix in the inputs array
        inputs[ii, :] = k
        ii += 1

    return inputs

def prot_encoding(protein_sequence):
    """
    Encodes a protein sequence into a numerical representation using an embedding matrix.

    Args:
        protein_sequence (str): The protein sequence to be encoded.

    Returns:
        numpy.ndarray: The encoded representation of the protein sequence.

    """
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
    nlength = len(protein_sequence)
    maxseq = 1000

    # Generates an array of dimension 1000x9, with all elements initialized to 0
    inputs = np.zeros([maxseq, 9])
    ii = 0

    # Iterates over each amino acid in the protein sequence
    for i in protein_sequence:
        # Finds the index of the amino acid in the amino_acids list
        j = amino_acids.index(i)
        # Retrieves the embedding matrix from the p_encoded array based on the amino acid index
        k = p_encoded[j, :]
        # Assigns the embedding matrix to the corresponding position in the inputs array
        inputs[ii, :] = k
        ii += 1

    return inputs
