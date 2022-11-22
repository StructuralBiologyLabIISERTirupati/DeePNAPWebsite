import numpy as np

def check_protein(sequence):
    sequence = sequence.replace(" ","")
    sequence = sequence.replace(",", "")
    sequence = sequence.replace("\n", "")
    sequence = sequence.replace("", "")
    if len(sequence)<1000:
        sequence = sequence.upper()
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        st = set(sequence)
        unrecognized = ""
        for i in st:
            if i not in amino_acids:
                if i in unrecognized:
                    pass
                else:
                    unrecognized += i
        if unrecognized !="":
            return "Unrecognized Amino Acid(s) {amino}. Amino acids must be one of ACDEFGHIKLMNPQRSTVWY ".format(amino = unrecognized), False
        else:
            return "Protein Sequence is Valid", True
    else:
        return "The Protein sequence is too long. It must not exceed 1000", False

def check_nacid(sequence):
    sequence = sequence.replace(" ","")
    sequence = sequence.replace(",", "")
    sequence = sequence.replace("\n", "")
    sequence = sequence.replace("", "")
    if len(sequence)<75:
        sequence = sequence.upper()
        allowed_nucleic_acids = "ATGCU"
        st = set(sequence)
        unrecognized = ""
        for i in st:
            if i not in allowed_nucleic_acids:
                if i in unrecognized:
                    pass
                else:
                    unrecognized += i
        if unrecognized !="":
            return "Unrecognized Nucleic Acid(s) {amino}. Nucleic acids must be one of ATGCU ".format(amino = unrecognized), False
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

