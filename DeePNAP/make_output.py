import numpy as np
from csv import writer


def make_output(output, n_acid, prot):
    """
    Converts the output value to physical quantities related to a binding equilibrium and writes them to a CSV file.

    Args:
        output (float): The logarithm (base 10) of the dissociation constant (KD).
        n_acid (int): Number of acid molecules.
        prot (str): The protein involved in the binding equilibrium.

    Returns:
        tuple: A tuple containing the calculated values of KD, KA, and delta G.

    """
    # Convert log base 10 (KD) to KD in Joules per mole
    kd = 10.0 ** output

    # Calculate association constant KA
    ka = kd ** (-1)

    # Calculate the standard free energy delta G degree = -RTln KA = RTln KD (T = 298.15 K) of the binding equilibrium
    R = 8.314
    T = 298.15
    G = R * T * np.log(kd)

    # Write the calculated values to a CSV file
    with open("app/output.csv", "w") as op_fyl:
        writer_obj = writer(op_fyl)
        writer_obj.writerow(["Kd", "Ka", "dG", "ddG", "N_acid", "Prot", "Label", "Mutation"]) # Writing the header
        writer_obj.writerow([kd, ka, G, 0, n_acid, prot, "W", ""]) # Writing the first row

    # Return the calculated values as a tuple
    return kd, ka, G


def mutant_output(mlkd, lkd, n_acid, prot, mutations):
    """
    Calculates various values related to mutant binding and writes them to a CSV file.

    Args:
        mlkd (float): Logarithm base 10 of mutant's KD value.
        lkd (float): Logarithm base 10 of KD value.
        n_acid (int): Number of acids.
        prot (str): Protein.
        mutations (str): Mutations.

    Returns:
        tuple: A tuple containing the calculated values mkd, mka, mG, and ddg.

    Note:
        The function assumes the existence of a CSV file named "app/output.csv" in append mode.
        The CSV file should have the following columns: Kd, Ka, dG, ddG, N_acid, Prot, Label, Mutation.
    """

    if lkd:
        # convert log base 10 (KD) to KD in Joules per mole
        mkd = 10.0 ** mlkd
        kd = 10.0 ** lkd

        # K association
        mka = mkd ** (-1)
        ka = kd ** (-1)

        # standard free energy delta G degree = -RTln KA = RTln KD (T = 298.15 K) of the binding equilibrium
        R = 8.314
        T = 298.15
        mG = R * T * np.log(mkd)
        G = R * T * np.log(kd)
        ddg = mG - G

        # Open the CSV file in append mode and write the calculated values
        with open("app/output.csv", "a") as op_fyl:
            writ = writer(op_fyl)
            mutations = mutations.replace(",", "+")  # Replace commas in mutations with plus signs
            writ.writerow([mkd, mka, mG, ddg, n_acid, prot, "M", mutations])  # Write the values to the CSV file

        return mkd, mka, mG, ddg
    else:
        # If lkd is not provided, return empty strings for all the calculated values
        return "", '', '', ''
