import numpy as np
from csv import writer


def make_output(output, n_acid, prot):
    # convert log base10 (KD) to KD in Joules per moules
    kd = 10.0 ** (output)

    # K association
    ka = (kd) ** (-1)

    # standard free energy delta G degree = -RTln KA = RTln KD (T = 298.15 K) of the binding equilibrium
    R = 8.314
    T = 298.15
    G = R * T * (np.log(kd))

    #kd = str(kd) + " mol/lit"
    #ka = str(ka) + " lit/mol"
    #G = str(G) + " Kcal/mol"
    #a = '''Kd Value = {kd} Ka Value = {ka} Delta G = {G} '''.format(kd = kd, ka = ka, G = G)
    with open("app/output.csv", "w") as op_fyl: #Writing to the file
        writ = writer(op_fyl)
        writ.writerow(["Kd", "Ka", "dG", "ddG", "N_acid", "Prot", "Label", "Mutation"]) #Writing the header
        writ.writerow([kd, ka, G, 0, n_acid, prot, "W", ""]) #Writing the first row


    return kd, ka, G
    #return a

def mutant_output(mlkd, lkd, n_acid, prot, mutations):
    if lkd:
        # convert log base10 (KD) to KD in Joules per moules
        mkd = 10.0 ** (mlkd)
        kd = 10.0 ** (lkd)
        # K association
        mka = (mkd) ** (-1)
        ka = kd ** (-1)

        # standard free energy delta G degree = -RTln KA = RTln KD (T = 298.15 K) of the binding equilibrium
        R = 8.314
        T = 298.15
        mG = R * T * (np.log(mkd))
        G = R * T * (np.log(kd))
        ddg = mG - G

        with open("app/output.csv", "a") as op_fyl:
            writ = writer(op_fyl)
            #writ.writerow(["Kd", "Ka", "dG", "ddG", "N_acid", "Prot", "Label", "Mutation"]) #Writing the header
            mutations = mutations.replace(",", "+")
            writ.writerow([mkd, mka, mG, ddg, n_acid, prot, "M", mutations])
        return mkd, mka, mG, ddg
        #a = '''Mutant's Kd value = {mkd} \n
         #   Mutant's Ka Value = {mka} \n
          #  Mutant's Delta G = {mG} \n
           # Delta Delta G = {ddg} '''.format(mkd = mkd, mka = mka, mG = mG, ddg = ddg)
        #return a
    else:
        return "",'','',''