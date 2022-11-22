'''Program to take inputs from the user in a web-framework and give an output based on a function'''

from flask import Flask, request, render_template, send_file
from DeePNAP.check_inputs import *
from DeePNAP.make_output import *
from tensorflow.keras import models
import keras
from keras.utils import np_utils
import numpy as np
from DeePNAP.data_encoding import *
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
from sklearn.model_selection import train_test_split
from tensorflow.keras import activations
from sklearn.utils import shuffle
from keras import backend as K
from keras.utils import np_utils
import pickle

# Protein: MAVRHERVAVRQERAVRTRQAIVRAAASVFDEYGFEAATVAEILSRASVTKGAMYFHFASKEELARGVLAEQTLHVAVPESGSKAQELVDLTMLVAHGMLHDPILRAGTRLALDQGAVDFSDANPFGEWGDICAQLLAEAQERGEVLPHVNPKKTGDFIVGCFTGLQAVSRVTSDRQDLGHRISVMWNHVLPSIVPASMLTWIETGEERIGKVAAAAEAAEAAEASEAASDE
# Nucleic Acid: GAGGCAAGCGAACCGCTCGGTTTGCTGAA
lrelu = Lambda(lambda x: tf.keras.activations.relu(x, alpha=0.2))
import numpy as np
from itertools import combinations
y=list(combinations('ABCDEF', 3))

STRING='ABCDEF'
pencoded=np.array([1,0,1,0,1,1])
for i in y:
   b6=np.zeros(6,dtype=int)
   for j in i:
     k=STRING.index(j)
     b6[k] = 1
   pencoded=np.vstack([pencoded,b6])

p_encoded = np.delete(pencoded, 0, 0)
#print(p_encoded)

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

import keras
from keras.utils import np_utils

def prot_encoding(protein_sequence):
  amino_acids=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S','T', 'V', 'W', 'Y', 'X']
  nlength=len(protein_sequence)
  maxseq=1000
 # generates a array of dimension 1000x22, with value 0
  inputs=np.zeros([maxseq,9])
  ii=0
  for i in protein_sequence:
  # gives the index of the amino acid in the sequence
    j=amino_acids.index(i)
  # takes in the position of amino acid in sequence and makes embedding matrix from it
    k=p_encoded[j,:]
    inputs[ii,:]= k
    ii += 1
  return inputs

def dr_encoding(dr_sequence):
  nucleo_base=['A', 'T', 'C', 'G', 'U']
  nlength=len(dr_sequence)
  maxseq=75
 # generates a array of dimension 1000x22, with value 0
  inputs=np.zeros([maxseq,5])
  ii=0
  for i in dr_sequence:
  # gives the index of the amino acid in the sequence
    j=nucleo_base.index(i)
  # takes in the position of amino acid in sequence and makes embedding matrix from it
#    k=drencoded[j,:]
    k=np_utils.to_categorical(j,num_classes=5)
    inputs[ii,:]= k
    ii += 1
  return inputs

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


model.load_weights('app/chkpointf1973.hdf5')


a = pwild_test
b = pmut_test
c = nacid_test

kdw_predict = model.predict({"pinp":a, "drinp":c,"pinp1":a, "drinp1":c})[0][0][0]
kdm_predict = model.predict({"pinp":a, "drinp":c,"pinp1":b, "drinp1":c})[1][0][0]
ddg_predict = model.predict({"pinp":a, "drinp":c,"pinp1":b, "drinp1":c})[2][0][0]

kdw_predict= 10.0**(kdw_predict) ###absolute value
kdm_predict= 10.0**(kdm_predict) ###absolute value
c=1.37295896724
ddg_predict=c*ddg_predict   ###in kcal/mol


print('Wild Kd : ', kdw_predict, "\n",
      'Mutant Kd :', kdm_predict, "\n",
      'Free Energy difference : ', ddg_predict)







def create_app(testing: bool = True):
        

    app = Flask('')

    app = Flask(__name__)


    @app.route('/')
    def home():
        return render_template("index.html")

    @app.route('/csv_output')
    def csv_output():
        path = "output.csv"
        return send_file(path, as_attachment = True)
    
    @app.route('/how_to_use')
    def how_to_use():
        return render_template("How to Use.html")
    
    @app.route('/about')
    def about_page():
        return render_template("About.html")


    @app.route("/predict", methods=["POST"])
    def predict():
        # Taking the input values from the program
        prot = str(request.form['Protein Sequence'])
        nacid = str(request.form['Nucleic Acid Sequence'])
        mutations = str(request.form["Mutations"])

        protein_check, prot_verify = check_protein(prot)
        nucleic_acid_check, naverify = check_nacid(nacid)
        if prot_verify and naverify:
            #Loading model from file
            #model = models.load_model("app/model.h5") #no longer needed with new model
            #Encoding protein and dna as inputs for model
            prot_str = np.zeros((1,1000,9))
            dr_str = np.zeros((1,75,5))
            prot_str[0, :, :] = prot_encoding(prot)
            dr_str[0, :, :] = dr_encoding(nacid)
            #Passing the protein and nucleic acid into the model
            
            output = model.predict({"pinp":prot_str, "drinp":dr_str,"pinp1":prot_str, "drinp1":dr_str})[0][0][0]
            print(mutations)
            # convert log base10 (KD) to KD in Joules per moules
            Kd_Value, Ka_value, G_value = make_output(output, nacid, prot)
            Kd_Value = str(Kd_Value) + " mol/lit"
            Ka_value = str(Ka_value) + " lit/mol"
            G_value = str(G_value) + " Kcal/mol"
            if mutations:
                #Applying the Mutation
                m_prot = mutate_protein(prot, mutations)
                #Encoding the Mutation as an input for the model
                mut_str = np.zeros((1,1000,9))
                mut_str[0,:,:] = prot_encoding(m_prot)
                #Passing the Mutation as an input into the model
                mut_kd = model.predict({"pinp":prot_str, "drinp":dr_str,"pinp1":mut_str, "drinp1":dr_str})[1][0][0]
                #Getting required values of the output from Model
                mKd_Value, mKa_value, mG_value, ddg = mutant_output(mut_kd, output, nacid, m_prot, mutations)
                mKd_Value = str(mKd_Value) + " mol/lit"
                mKa_value = str(mKa_value) + " lit/mol"
                mG_value = str(mG_value) + " Kcal/mol"
                ddg = str(ddg) + " Kcal/mol"
            else:
                mKd_Value, mKa_value, mG_value, ddg = "","","",""

        else:
            Kd_Value, Ka_value, G_value = "", "", ""
            mKd_Value, mKa_value, mG_value, ddg = "", "", "", ""

        return render_template("index.html",
                            protein_check=protein_check,
                            nucleic_acid_check=nucleic_acid_check,
                            Kd_Value = Kd_Value,
                            Ka_value = Ka_value,
                            G_value = G_value,
                            mKd_Value = mKd_Value,
                            mKa_value = mKa_value,
                            mG_value = mG_value,
                            ddg = ddg
                            )

    return app 




