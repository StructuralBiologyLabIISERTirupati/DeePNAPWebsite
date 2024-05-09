'''Program to take inputs from the user in a web-framework and give an output based on a function'''

from flask import Flask, request, render_template, send_file

from tensorflow.keras import models
import keras
import numpy as np
from tensorflow.keras.utils import to_categorical
# required imports
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from keras.layers import Input, Conv2D, Lambda
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Bidirectional, LSTM, Reshape

from tensorflow.keras import activations

from keras import backend as K
import keras

pwd = ""


def check_protein(sequence):
    """
    Check if a protein sequence is valid.

    Args:
        sequence (str): The protein sequence to be checked.

    Returns:
        tuple: A tuple containing a message string and a boolean value.
               The message string describes the result of the validation,
               and the boolean value indicates whether the sequence is valid or not.
    """

    # Remove unnecessary characters from the sequence
    sequence = sequence.replace(" ","")  # Remove spaces
    sequence = sequence.replace(",", "")  # Remove commas
    sequence = sequence.replace("\n", "")  # Remove newlines
    sequence = sequence.replace("", "")  # Remove empty strings
    sequence = sequence.replace("-", "")  # Remove dashes
    sequence = sequence.replace(">", "")  # Remove ">" symbols

    if len(sequence) < 1000:
        sequence = sequence.upper()  # Convert the sequence to uppercase
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # Valid amino acids
        st = set(sequence)  # Get the unique characters in the sequence
        unrecognized = ""

        # Check for unrecognized amino acids
        for i in st:
            if i not in amino_acids:
                if i in unrecognized:
                    pass
                else:
                    unrecognized += i

        if unrecognized != "":
            # If unrecognized amino acids are found, return an error message
            return (
                "Unrecognized Amino Acid(s) {amino}. Amino acids must be one of ACDEFGHIKLMNPQRSTVWY".format(amino=unrecognized),
                False,
            )
        else:
            # If all amino acids are recognized, return a success message
            return "Protein Sequence is Valid", True
    else:
        # If the sequence is too long, return an error message
        return "The Protein sequence is too long. It must not exceed 1000", False


def check_nacid(sequence):
    """
    Checks the validity of a nucleic acid sequence.

    Parameters:
    - sequence (str): The nucleic acid sequence to be checked.

    Returns:
    - tuple: A tuple containing a message and a boolean indicating the validity.
        - If the sequence is valid, the message is "Nucleic Acid Sequence is Valid" and the boolean is True.
        - If the sequence is invalid, the message indicates the specific error and the boolean is False.
    """

    # Remove unwanted characters from the sequence
    sequence = sequence.replace(" ","")
    sequence = sequence.replace(",", "")
    sequence = sequence.replace("\n", "")
    sequence = sequence.replace("", "")
    sequence = sequence.replace("-", "")

    # Check the length of the sequence
    if len(sequence) < 75:
        sequence = sequence.upper()
        allowed_nucleic_acids = "ATGCU"
        st = set(sequence)
        unrecognized = ""

        # Check each character in the sequence
        for i in st:
            if i not in allowed_nucleic_acids:
                if i in unrecognized:
                    pass
                else:
                    unrecognized += i

        # If unrecognized nucleic acids are found, return an error message
        if unrecognized != "":
            return "Unrecognized Nucleic Acid(s) {amino}. Nucleic acids must be one of ATGCU ".format(amino=unrecognized), False
        else:
            return "Nucleic Acid Sequence is Valid", True
    else:
        return "The Nucleic sequence is too long. It must not exceed 75", False



def predict_wild(prot1, na1, makefile=True):
    """
    Predicts wild-type protein properties and writes the results to a CSV file if makefile is True.

    Parameters:
        prot1 (str): The wild-type protein sequence.
        na1 (str): The nucleic acid sequence.
        makefile (bool, optional): Specifies whether to create a CSV file with the predicted results. Defaults to True.

    Returns:
        tuple: A tuple containing the predicted Kd (dissociation constant), Ka (association constant),
               and G (Gibbs free energy) values.
    """
    global model  # Access the global variable 'model'
    from csv import writer  # Import the 'writer' function from the 'csv' module
    prot_str = np.zeros((1, 1000, 9))  # Create a 3D numpy array filled with zeros
    dr_str = np.zeros((1, 75, 5))  # Create another 3D numpy array filled with zeros
    prot1 = clean_sequence_protein(prot1)
    prot_str[0, :, :] = prot_encoding(prot1)  # Encode the wild-type protein sequence
    dr_str[0, :, :] = dr_encoding(na1)  # Encode the nucleic acid sequence

    chkfiles=['checkpoint_files/17_chkpointf1996.hdf5','checkpoint_files/37_chkpointf1959.hdf5', 'checkpoint_files/4_chkpointf0966.hdf5','checkpoint_files/2_chkpointf2376.hdf5','checkpoint_files/38_chkpointf1374.hdf5']
    kdw1_predict = np.zeros((1, 1))
    #kdm1_predict = np.zeros((1, 1))
    ddg1_predict = np.zeros((1, 1))


    # Iterate over the checkpoint files
    for i in range(5):
        model.load_weights(chkfiles[i])
        kdw_pred=model.predict({"pinp":prot_str, "drinp":dr_str, "pinp1":prot_str, "drinp1":dr_str}, verbose=0)[0]
        ddg_pred=model.predict({"pinp":prot_str, "drinp":dr_str, "pinp1":prot_str, "drinp1":dr_str}, verbose=0)[2]

        kdw_predict += kdw_pred
        kdm_predict += kdm_pred
        ddg_predict += ddg_pred

    kdw1 = kdw_predict
    kdw_predict = 10.0**(kdw1/5)  # Calculate the absolute value of kdw_predict

    c = 1.37295896724
    #ddg_predict=c*(kdw1_predict-kdm1_predict)/4.0   ###in kcal/mol

    kd = kdw_predict
    ka = 1 / kd
    G = c * (kdw1/5)
    kd, ka, G = kd[0][0], ka[0][0], G[0][0]
    #dgm = c*(kdm1_predict/4.0)
    #ddg = -c*(kdm1_predict/4)+c*(kdw1_predict/4.0)

    if makefile == True:
        # Write the predicted results to a CSV file
        with open("output.csv", "w") as op_fyl:
            writ = writer(op_fyl)  # Create a CSV writer object
            writ.writerow(["Kd", "Ka", "dG", "ddG", "N_acid", "Prot", "Label", "Mutation"])  # Write the header
            writ.writerow([kd, ka, G, 0, na1, prot1, "W", ""])  # Write the first row

    return kd, ka, G

def predict_mutant(prot1, na1, mutations):
    """Predicts the properties of a mutant protein based on the given inputs.

    Args:
        prot1 (str): The wild-type protein sequence.
        na1 (str): The nucleic acid sequence.
        mutations (str): The mutations made to the protein sequence.

    Returns:
        tuple: A tuple containing the predicted properties of the wild-type and mutant proteins.
               The tuple contains the following values:
               - kd (float): The dissociation constant of the wild-type protein.
               - ka (float): The association constant of the wild-type protein.
               - G (float): The free energy change of the wild-type protein.
               - mkd (float): The dissociation constant of the mutant protein.
               - mka (float): The association constant of the mutant protein.
               - mG (float): The free energy change of the mutant protein.
               - ddg (float): The difference in free energy between the wild-type and mutant proteins.
    """
    global model
    from csv import writer
    prot2 = mutate_protein(prot1, mutations)
    prot1 = clean_sequence_protein(prot1)
    prot2 = clean_sequence_protein(prot2)

    # Encoding the protein and nucleic acid sequences
    prot_str = np.zeros((1, 1000, 9))
    dr_str = np.zeros((1, 75, 5))
    prot_str[0, :, :] = prot_encoding(prot1)
    dr_str[0, :, :] = dr_encoding(na1)
    mut_str = np.zeros((1, 1000, 9))
    mut_str[0, :, :] = prot_encoding(prot2)

    # Checkpoint files for the model
    chkfiles=['checkpoint_files/17_chkpointf1996.hdf5','checkpoint_files/37_chkpointf1959.hdf5', 'checkpoint_files/4_chkpointf0966.hdf5','checkpoint_files/2_chkpointf2376.hdf5','checkpoint_files/38_chkpointf1374.hdf5']
    kdw1_predict = np.zeros((1, 1))
    kdm1_predict = np.zeros((1, 1))
    ddg1_predict = np.zeros((1, 1))

    # Iterate over the checkpoint files
    # Iterate over the checkpoint files
    for i in range(5):
        model.load_weights(chkfiles[i])
        kdw_predict=model.predict({"pinp":prot_str, "drinp":dr_str, "pinp1":mut_str, "drinp1":dr_str}, verbose=0)[0]
        kdm_predict=model.predict({"pinp":prot_str, "drinp":dr_str, "pinp1":mut_str, "drinp1":dr_str}, verbose=0)[1]
        ddg_predict=model.predict({"pinp":prot_str, "drinp":dr_str, "pinp1":mut_str, "drinp1":dr_str}, verbose=0)[2]

        kdw1_predict += kdw_predict
        kdm1_predict += kdm_predict
        ddg1_predict += ddg_predict

    kdw_predict = 10.0**(kdw1_predict/5)  # Calculate the absolute value of kdw_predict

    # Calculate the predicted values
    kdw_predict = 10.0 ** (kdw1_predict / 5.0)
    kdm_predict = 10.0 ** (kdm1_predict / 5.0)
    c = 1.37295896724
    kd = kdw_predict
    ka = 1 / kd
    G = c * (kdw1_predict / 5.0)
    kd, ka, G = kd[0][0], ka[0][0], G[0][0]

    mkd = kdm_predict
    mka = 1 / mkd
    mG = c * (kdm1_predict / 5.0)
    mkd, mka, mG = mkd[0][0], mka[0][0], mG[0][0]
    #ddg = 1.37295896724 * ddg1_predict
    ddg = c*(-kdw1_predict + kdm1_predict)/5.0   ###in kcal/mol

    ddg = ddg[0][0]

    mutations = mutations.replace(",", "+")
    data = [["Kd", "Ka", "dG", "ddG", "N_acid", "Prot", "Label", "Mutation"],
            [kd, ka, G, 0, na1, prot1, "W", ""],
            [mkd, mka, mG, ddg, na1, prot2, "M", mutations]]

    df = pd.DataFrame(data[1:], columns=data[0])

    df.to_csv("output.csv", index=False)

    return kd, ka, G, mkd, mka, mG, ddg


def predict_fasta(prot_fasta, nacid_fasta):
    """
    Predicts values for proteins and nucleic acids based on input FASTA sequences.

    Args:
        prot_fasta (str): Path to the protein FASTA file.
        nacid_fasta (str): Path to the nucleic acid FASTA file.

    Returns:
        str: A message indicating the validity of the inputs or an error message.

    Raises:
        FileNotFoundError: If the FASTA files are not found.
    """
    # Parse protein and nucleic acid FASTA files
    prot_fasta = parse_fasta(prot_fasta)
    nacid_fasta = parse_fasta(nacid_fasta)

    # Check if the number of proteins and nucleic acids entered is equal
    if len(prot_fasta) != len(nacid_fasta):
        return "Number of Proteins and Nucleic Acids entered is not Equal"

    kd_vals = []  # List to store Kd values
    ka_vals = []  # List to store Ka values
    G_vals = []   # List to store Delta G values

    # Iterate over each pair of protein and nucleic acid sequences
    for i in range(len(prot_fasta)):
        # Check if the length of sequences exceeds the model's limits
        if len(prot_fasta[i]) > 1000 or len(nacid_fasta[i]) > 75:
            error_message = """One or more of the sequences are longer than what the model can support.
               Ensure that the Protein Sequence is not longer than 1000 and
               Nucleic Acid sequence is not longer than 75"""
            return error_message

        # Predict Kd, Ka, and Delta G values for the current pair of sequences
        kd, ka, G = predict_wild(prot_fasta[i], nacid_fasta[i], makefile=False)
        kd_vals.append(kd)
        ka_vals.append(ka)
        G_vals.append(G)

    # Create a dictionary with the results
    deets = {
        "Protein": prot_fasta,
        "Nucleic Acid": nacid_fasta,
        "Kd": kd_vals,
        "Ka": ka_vals,
        "Delta G": G_vals
    }

    # Create a pandas DataFrame from the dictionary and save it to a CSV file
    df = pd.DataFrame(deets)
    df.to_csv("fasta_output.csv")

    return "The Inputs are Valid"

def parse_fasta(fasta_sequence):
    """
    Parses a FASTA sequence and returns a list of cleaned sequences.

    Args:
        fasta_sequence (str): The input FASTA sequence.

    Returns:
        list: A list of cleaned sequences.

    """

    fasta_sequence = fasta_sequence.split(">")  # Splitting over the ">" symbol
    fasta_sequence.remove("")  # Removes the empty string the FASTA sequence begins with.

    for i in range(len(fasta_sequence)):
        sequence = fasta_sequence[i]  # Going over the sequences one by one
        sequence = sequence.split("\n")  # Splitting the sequence into lines
        sequence[0] = ""  # Setting the first line of the sequence to be an empty string.
        # This ignores the line which began with a ">"
        sequence = "".join(sequence)  # Joining the remaining lines

        sequence = clean_sequence(sequence)  # Cleaning the sequence
        fasta_sequence[i] = sequence  # Updating the list with the cleaned sequence

    return fasta_sequence



def round_off(val):
    """Rounds off a given value to four decimal places if it contains an 'e' character in its string representation.

    Args:
        val (float or int): The value to be rounded off.

    Returns:
        str: The rounded value as a string.
    """
    if "e" in str(val):  # Checks if the string representation of val contains 'e'
        val = str(val).split("e")  # Splits the string at 'e' to separate the mantissa and exponent
        num = float(val[0])  # Extracts the mantissa and converts it to a float
        num = np.round(num, 4)  # Rounds off the mantissa to four decimal places using numpy
        val = str(num) + "e" + val[1]  # Combines the rounded mantissa and original exponent
    else:
        val = float(val)  # Converts the value to a float
        num = np.round(val, 4)  # Rounds off the value to four decimal places using numpy
        val = str(num)  # Converts the rounded value back to a string
    return val


def clean_sequence(sequence):
    """Cleans a DNA sequence by removing any non-alphabetic characters and a leading '>' symbol if present.

    Args:
        sequence (str): The DNA sequence to be cleaned.

    Returns:
        str: The cleaned DNA sequence.
    """
    if sequence[0] == ">":  # Checks if the sequence starts with '>'
        sequence = sequence.split("\n")  # Splits the sequence into lines at each newline character
        sequence[0] = ""  # Removes the first line (which starts with '>')
        sequence = "".join(sequence)  # Joins the lines back together into a single string
    nseq = ""
    for i in sequence:  # Iterates over each character in the sequence
        if i in ['A', "T", "G", "C", "U"]:  # Checks if the character is alphabetic
            nseq = nseq + i  # Appends the alphabetic character to the cleaned sequence
    return nseq  # Returns the cleaned DNA sequence

def clean_sequence_protein(sequence):
    """Cleans a Protein sequence by removing any non-alphabetic characters and a leading '>' symbol if present.

    Args:
        sequence (str): The Protein sequence to be cleaned.

    Returns:
        str: The cleaned Protein sequence.
    """
    if sequence[0] == ">":  # Checks if the sequence starts with '>'
        sequence = sequence.split("\n")  # Splits the sequence into lines at each newline character
        sequence[0] = ""  # Removes the first line (which starts with '>')
        sequence = "".join(sequence)  # Joins the lines back together into a single string
    nseq = ""
    for i in sequence:  # Iterates over each character in the sequence
        if i in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']:  # Checks if the character is alphabetic
            nseq = nseq + i  # Appends the alphabetic character to the cleaned sequence
    return nseq  # Returns the cleaned Protein sequence







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
        k = to_categorical(j, num_classes=5)

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


# Protein: MAVRHERVAVRQERAVRTRQAIVRAAASVFDEYGFEAATVAEILSRASVTKGAMYFHFASKEELARGVLAEQTLHVAVPESGSKAQELVDLTMLVAHGMLHDPILRAGTRLALDQGAVDFSDANPFGEWGDICAQLLAEAQERGEVLPHVNPKKTGDFIVGCFTGLQAVSRVTSDRQDLGHRISVMWNHVLPSIVPASMLTWIETGEERIGKVAAAAEAAEAAEASEAASDE
# Nucleic Acid: GAGGCAAGCGAACCGCTCGGTTTGCTGAA

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


pwild_test=np.zeros((1,1000,9))
pmut_test=np.zeros((1,1000,9))
nacid_test=np.zeros((1,75,5))

ftr1='MTWLPLNPIPLKDRVSMIFLQYGQIDVIDGAFVLIDKTGIRTHIPVGSVACIMLEPGTRVSHAAVRLAAQVGTLLVWVGEAGVRVYASGQPGGARSDKLLYQAKLALDEDLRLKVVRKMFELRFGEPAPARRSVEQLRGIEGSRVRATYALLAKQYGVTWNGRRYDPKDWEKGDTINQCISAATSCLYGVTEAAILAAGYAPAIGFVHTGKPLSFVYDIADIIKFDTVVPKAFEIARRNPGEPDREVRLACRDIFRSSKTLAKLIPLIEDVLAAGEIQPPAPPEDAQPVAIPLPVSLGDAGHRSS'
ftr2='MTWLPLNPIPLKDRVSMIFLQYGQIDVIDGAFVLIDKTGIRTHIPVGSVACIMLEPGTRVSHAAVRLAAQVGTLLVWVGEAGVRVYASGQPGGARSDKLLYQAKLALDEDLRLKVVRKMFELRFGEPAPARRSVEQLRGIEGSRVRATYALLAKQYGVTWNGRRYDPKDWEKGDTINQCISAATSCLYGVTEAAILAAGYAPAIGFVHTGKPLSFVYDIADIIKFDTVVPKAFEIARRNPGEPDREVRLACRDIFRSSKTLAKLIPLIEDVLAAGEIQPPAPPEDAQPVAIPLPVSLGDAGHRSS'

ftr3='ATTTACTACTCGTTCTGGTGTTTCTCGT'

ftr1=ftr1.upper()
ftr2=ftr2.upper()
ftr3=ftr3.upper()

pwild_test[0,:,:]=prot_encoding(ftr1)
pmut_test[0,:,:]=prot_encoding(ftr2)

nacid_test[0,:,:]=dr_encoding(ftr3)


lrelu = lambda x: tf.keras.layers.LeakyReLU(alpha=0.1)(x)

protein_input = keras.Input(shape=(1000, 9, 1), name="pinp")

dr_input = keras.Input(shape=(75, 5, 1), name="drinp")

protein_input1 = keras.Input(shape=(1000, 9, 1), name="pinp1")

dr_input1 = keras.Input(shape=(75, 5, 1), name="drinp1")

npf=48
ndf=32
nft=6
np1=168
np2=168
nd1=38
nd2=38

p01 = layers.ZeroPadding2D(padding=((0, 8),(0,0)))
p1=p01(protein_input)
p02 = layers.Conv2D(npf, (nft,9), strides=(nft, 1), activation='relu', kernel_regularizer=keras.regularizers.l1(5e-4))
p1=p02(p1)
p03=layers.Reshape((np1,npf,1), input_shape=(np1,1,npf))
p1=p03(p1)
p04=layers.MaxPooling2D(pool_size=(1,npf))
p1=p04(p1)
p05=layers.Flatten()
p1=p05(p1)

p11 = layers.ZeroPadding2D(padding=((3, 5),(0,0)))
p2=p11(protein_input)
p12 = layers.Conv2D(npf, (nft,9), strides=(nft, 1), activation='relu', kernel_regularizer=keras.regularizers.l1(5e-4))
p2=p12(p2)
p13=layers.Reshape((np2,npf,1), input_shape=(np2,1,npf))
p2=p13(p2)
p14=layers.MaxPooling2D(pool_size=(1,npf))
p2=p14(p2)
p15=layers.Flatten()
p2=p15(p2)

d01 = layers.ZeroPadding2D(padding=((0, 1),(0,0)))
d1=d01(dr_input)
d02 = layers.Conv2D(ndf, (2,5), strides=(2, 1), activation='relu', kernel_regularizer=keras.regularizers.l1(5e-4))
d1=d02(d1)
d03=layers.Reshape((nd1,ndf,1), input_shape=(nd1,1,ndf))
d1=d03(d1)
d04=layers.MaxPooling2D(pool_size=(1,ndf))
d1=d04(d1)
d05=layers.Flatten()
d1=d05(d1)

d11 = layers.ZeroPadding2D(padding=((1, 0),(0,0)))
d2=d11(dr_input)
d12 = layers.Conv2D(ndf, (2,5), strides=(2, 1), activation='relu', kernel_regularizer=keras.regularizers.l1(5e-4))
d2=d12(d2)
d13=layers.Reshape((nd2,ndf,1), input_shape=(nd2,1,ndf))
d2=d13(d2)
d14=layers.MaxPooling2D(pool_size=(1,ndf))
d2=d14(d2)
d15=layers.Flatten()
d2=d15(d2)

n_n=96
n_n1=32
pdnn1=layers.concatenate([p1, d1])

pdnn01=layers.Dense(n_n, activation="relu")
pdnn1=pdnn01(pdnn1)
pdnn02=layers.Dropout(0.5, input_shape=(n_n,))
pdnn1=pdnn02(pdnn1)
pdnn03=layers.Dense(n_n1,kernel_regularizer=keras.regularizers.l1(1e-3))
pdnn1=pdnn03(pdnn1)

pdnn2=layers.concatenate([p1, d2])

pdnn11=layers.Dense(n_n, activation="relu")
pdnn2=pdnn11(pdnn2)
pdnn12=layers.Dropout(0.5, input_shape=(n_n,))
pdnn2=pdnn12(pdnn2)
pdnn13=layers.Dense(n_n1,kernel_regularizer=keras.regularizers.l1(1e-3))
pdnn2=pdnn13(pdnn2)

pdnn3=layers.concatenate([p2, d1])

pdnn21=layers.Dense(n_n, activation="relu")
pdnn3=pdnn21(pdnn3)
pdnn22=layers.Dropout(0.5, input_shape=(n_n,))
pdnn3=pdnn22(pdnn3)
pdnn23=layers.Dense(n_n1,kernel_regularizer=keras.regularizers.l1(1e-3))
pdnn3=pdnn23(pdnn3)

pdnn4=layers.concatenate([p2, d2])
pdnn31=layers.Dense(n_n, activation="relu")
pdnn4=pdnn31(pdnn4)
pdnn32=layers.Dropout(0.5, input_shape=(n_n,))
pdnn4=pdnn32(pdnn4)
pdnn33=layers.Dense(n_n1,kernel_regularizer=keras.regularizers.l1(1e-3))
pdnn4=pdnn33(pdnn4)

pdnn_0=layers.concatenate([pdnn1, pdnn2, pdnn3, pdnn4])

pdnn41=layers.Dense(256, activation=lrelu)
pdnn_01=pdnn41(pdnn_0)
pdnn42=layers.Dropout(0.5, input_shape=(256,))
pdnn_01=pdnn42(pdnn_01)
pdnn43=layers.Dense(128, activation=lrelu)
pdnn_01=pdnn43(pdnn_01)
pdnn44=layers.Dropout(0.5, input_shape=(128,))
pdnn_01=pdnn44(pdnn_01)
pdnn_01=layers.Add()([pdnn_01,pdnn_0])
pdnn45=layers.Dense(64, activation=lrelu, kernel_regularizer=keras.regularizers.l1(1e-4))
pdnn_01=pdnn45(pdnn_01)
pdnn46=layers.Dense(1)
pdnn_01=pdnn46(pdnn_01)
pdnn_01 = layers.Lambda(lambda x: x, name="out0")(pdnn_01)

p1_1=p01(protein_input1)
p1_1=p02(p1_1)
p1_1=p03(p1_1)
p1_1=p04(p1_1)
p1_1=p05(p1_1)

p2_1=p11(protein_input1)
p2_1=p12(p2_1)
p2_1=p13(p2_1)
p2_1=p14(p2_1)
p2_1=p15(p2_1)

d1_1=d01(dr_input1)
d1_1=d02(d1_1)
d1_1=d03(d1_1)
d1_1=d04(d1_1)
d1_1=d05(d1_1)

d2_1=d11(dr_input1)
d2_1=d12(d2_1)
d2_1=d13(d2_1)
d2_1=d14(d2_1)
d2_1=d15(d2_1)

pdnn1_1=layers.concatenate([p1_1, d1_1])
pdnn1_1=pdnn01(pdnn1_1)
pdnn1_1=pdnn02(pdnn1_1)
pdnn1_1=pdnn03(pdnn1_1)

pdnn2_1=layers.concatenate([p1_1, d2_1])
pdnn2_1=pdnn11(pdnn2_1)
pdnn2_1=pdnn12(pdnn2_1)
pdnn2_1=pdnn13(pdnn2_1)

pdnn3_1=layers.concatenate([p2_1, d1_1])
pdnn3_1=pdnn21(pdnn3_1)
pdnn3_1=pdnn22(pdnn3_1)
pdnn3_1=pdnn23(pdnn3_1)

pdnn4_1=layers.concatenate([p2_1, d2_1])
pdnn4_1=pdnn31(pdnn4_1)
pdnn4_1=pdnn32(pdnn4_1)
pdnn4_1=pdnn33(pdnn4_1)

pdnn_1=layers.concatenate([pdnn1_1, pdnn2_1, pdnn3_1, pdnn4_1])

pdnn_11=pdnn41(pdnn_1)
pdnn_11=pdnn42(pdnn_11)
pdnn_11=pdnn43(pdnn_11)
pdnn_11=pdnn44(pdnn_11)
pdnn_11=layers.Add()([pdnn_11,pdnn_1])
pdnn_11=pdnn45(pdnn_11)
pdnn_11=pdnn46(pdnn_11)
pdnn_11 = layers.Lambda(lambda x: x, name="out1")(pdnn_11)

pdnn_22 = layers.Subtract(name="out2")([pdnn_01,pdnn_11])

model = keras.Model(
    inputs=[protein_input, dr_input, protein_input1, dr_input1],
    outputs=[pdnn_01, pdnn_11, pdnn_22])

chkfiles=['checkpoint_files/17_chkpointf1996.hdf5','checkpoint_files/37_chkpointf1959.hdf5', 'checkpoint_files/4_chkpointf0966.hdf5','checkpoint_files/2_chkpointf2376.hdf5','checkpoint_files/38_chkpointf1374.hdf5']

kdw_predict = 0
kdm_predict = 0
ddg_predict = 0

a = pwild_test
b = pmut_test
c = nacid_test

for i in range(5):
  model.load_weights(chkfiles[i])
  kdw_pred=model.predict({"pinp":a, "drinp":c, "pinp1":a, "drinp1":c}, verbose=0)[0]
  kdm_pred=model.predict({"pinp":a, "drinp":c, "pinp1":b, "drinp1":c}, verbose=0)[1]
  ddg_pred=model.predict({"pinp":a, "drinp":c, "pinp1":b, "drinp1":c}, verbose=0)[2]

  kdw_predict += kdw_pred
  kdm_predict += kdm_pred
  ddg_predict += ddg_pred

print("Kd(wild) =",10.0**(kdw_predict/5))
print("Kd(mutant) =",10.0**(kdm_predict/5))
print("ddG (kcal/mol) =",ddg_predict/5*1.37295896724)




def create_app(testing: bool = True):
    """
    This script defines multiple routes for a Flask web application.
    Each route corresponds to a specific URL endpoint and returns a specific HTML template.

    Routes:
    - '/' : Returns the "About.html" template.
    - '/csv_output' : Downloads the "output.csv" file.
    - '/download_fasta' : Downloads the "fasta_output.csv" file.
    - '/how_to_use' : Returns the "How to Use.html" template.
    - '/about' : Returns the "About.html" template.
    - '/example' : Returns the "example.html" template.
    - '/fasta_input' : Returns the "fasta.html" template.
    - '/input_page' : Returns the "index.html" template.
    """


    #app = Flask('')

    app = Flask(__name__)



    @app.route('/')
    def home():
        """
        Renders the "About.html" template and returns it as the response for the '/' endpoint.
        """
        return render_template("About.html")


    @app.route('/csv_output')
    def csv_output():
        """
        Downloads the "output.csv" file when the '/csv_output' endpoint is accessed.
        The file is sent as an attachment to the client's browser.
        """
        path = "output.csv"
        return send_file(path, as_attachment=True)


    @app.route('/download_fasta')
    def download_fasta():
        """
        Downloads the "fasta_output.csv" file when the '/download_fasta' endpoint is accessed.
        The file is sent as an attachment to the client's browser.
        """
        path = "fasta_output.csv"
        return send_file(path, as_attachment=True)


    @app.route('/how_to_use')
    def how_to_use():
        """
        Renders the "How to Use.html" template and returns it as the response for the '/how_to_use' endpoint.
        """
        return render_template("How to Use.html")


    @app.route('/about')
    def about_page():
        """
        Renders the "About.html" template and returns it as the response for the '/about' endpoint.
        """
        return render_template("About.html")


    @app.route("/example")
    def load_example():
        """
        Renders the "example.html" template and returns it as the response for the '/example' endpoint.
        """
        return render_template("example.html")


    @app.route("/fasta_input")
    def fasta_input():
        """
        Renders the "fasta.html" template and returns it as the response for the '/fasta_input' endpoint.
        """
        return render_template("fasta.html")


    @app.route("/input_page")
    def input_page():
        """
        Renders the "index.html" template and returns it as the response for the '/input_page' endpoint.
        """
        return render_template("index.html")

    @app.route("/fasta", methods=["POST"])
    def fasta():
        """
        Handle the POST request to process FASTA sequences.

        This function receives a POST request with protein and nucleic acid sequences in FASTA format.
        It calls the `predict_fasta` function to validate the input sequences and generate an error message.
        If the sequences are valid, it sets a download link for the generated file.
        Finally, it renders the "fasta.html" template with the error message and download link.

        Returns:
            A rendered HTML template with the error message and download link.
        """
        # Get the protein and nucleic acid sequences from the request form
        prot = str(request.form['Protein Sequence fasta'])
        nacid = str(request.form['Nucleic Acid Sequence fasta'])

        # Call the predict_fasta function to validate the input sequences and generate an error message
        error_message = predict_fasta(prot, nacid)

        # Initialize the download link as an empty string
        download_link = ""

        # If the error message indicates that the inputs are valid, set a download link
        if error_message == "The Inputs are Valid":
            download_link = "Click here to download the File"

        # Render the "fasta.html" template with the error message and download link
        return render_template("fasta.html",
                            generate_output=error_message,
                            download_link=download_link)




    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Predict function for a web application that takes input values from the user,
        performs calculations, and returns predictions.

        Returns:
            str: Rendered HTML template with the predicted values and checks for the input sequences.
        """
        # Taking the input values from the program
        prot = str(request.form['Protein Sequence'])  # Get the protein sequence from the form data
        nacid = str(request.form['Nucleic Acid Sequence'])  # Get the nucleic acid sequence from the form data
        mutations = str(request.form["Mutations"])  # Get the mutations from the form data

        # Check the validity of protein sequence and nucleic acid sequence
        protein_check, prot_verify = check_protein(prot)
        nucleic_acid_check, naverify = check_nacid(nacid)

        if prot_verify and naverify:  # If both protein and nucleic acid sequences are valid
            # Clean the sequences (remove unwanted characters)
            prot = clean_sequence(prot)
            nacid = clean_sequence(nacid)

            if mutations:  # If mutations are provided, predict for the mutant
                # Perform the prediction for the mutant
                Kd_value, Ka_value, G_value, mKd_value, mKa_value, mG_value, ddg = predict_mutant(prot, nacid, mutations)

                # Round off the values and add units to the predictions
                Kd_value = round_off(Kd_value) + " mol/lit"
                Ka_value = round_off(Ka_value) + " lit/mol"
                G_value = round_off(G_value) + " Kcal/mol"

                mKd_value = round_off(mKd_value) + " mol/lit"
                mKa_value = round_off(mKa_value) + " lit/mol"
                mG_value = round_off(mG_value) + " Kcal/mol"
                ddg = str(ddg) + " Kcal/mol"

            else:  # If no mutations are provided, predict for the wild type
                # Perform the prediction for the wild type
                Kd_value, Ka_value, G_value = predict_wild(prot, nacid)
                mKd_value, mKa_value, mG_value, ddg = "", "", "", ""  # Set mutant values as empty strings

                # Round off the values and add units to the predictions
                Kd_value = round_off(Kd_value) + " mol/lit"
                Ka_value = round_off(Ka_value) + " lit/mol"
                G_value = round_off(G_value) + " Kcal/mol"

        else:  # If either protein or nucleic acid sequences are invalid
            # Set all predictions as empty strings
            Kd_value, Ka_value, G_value = "", "", ""
            mKd_value, mKa_value, mG_value, ddg = "", "", "", ""

        # Render the HTML template with the predicted values and checks for the input sequences
        return render_template("index.html",
                            protein_check=protein_check,
                            nucleic_acid_check=nucleic_acid_check,
                            Kd_Value=Kd_value,
                            Ka_value=Ka_value,
                            G_value=G_value,
                            mKd_Value=mKd_value,
                            mKa_value=mKa_value,
                            mG_value=mG_value,
                            ddg=ddg
                            )


    return app 




