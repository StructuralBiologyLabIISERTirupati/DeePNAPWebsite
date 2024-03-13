import numpy as np

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




if __name__ == "__main__":
    print(check_protein('''MARQLRAEQTRATIIGAAADLFDRRGYESTTLSEIVAHAGVTKGALYFHFAAKEDLAHAILEIQSRTSRRLA
KDLDGRGYSSLEALMRLTFGMARLCVQGPVLRAGLRLATAGVPVRPPLPHPFTEWREIATSRLLDAVRQS
DVHQDIDVDSVAHTLVCSVVGTRVVGGTLEPAGREPRRLAEMWYILIRGMVPVTRRARYVTLAARLEQE
TGTA'''))



'''MARQLRAEQTRATIIGAAADLFDRRGYESTTLSEIVAHAGVTKGALYFHFAAKEDLAHAILEIQSRTSRRLA
KDLDGRGYSSLEALMRLTFGMARLCVQGPVLRAGLRLATAGVPVRPPLPHPFTEWREIATSRLLDAVRQS
DVHQDIDVDSVAHTLVCSVVGTRVVGGTLEPAGREPRRLAEMWYILIRGMVPVTRRARYVTLAARLEQE
TGTA
ACATACGGGACGCCCCGTTTAT'''

