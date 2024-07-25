import numpy as np

def reverse_complement(sequence):
    """
    Reverse complement the input nucleotide sequence.

    Args:
    - sequence (str): Input nucleotide sequence.

    Returns:
    - str: Reverse complemented nucleotide sequence.
    """
    # Reverse the sequence
    reversed_sequence = sequence[::-1]

    # Map each nucleotide to its complement
    complement_mapping = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', '-': '-'}

    # Complement each nucleotide
    complemented_sequence = ''.join(complement_mapping.get(nuc, nuc) for nuc in reversed_sequence)

    return complemented_sequence

def one_hot_encoding(array, num_class: int):
    '''
        只对 A,C,G,T,-,[PAD] encoding
        A: [1, 0, 0, 0, 0], ..., [PAD]: [0, 0, 0, 0, 0]
    '''
    output = []
    mapping = np.eye(num_class)
    for value in array: # type: ignore
        if value == 0:
            output.append(np.zeros(num_class))
        else:
            output.append(np.eye(num_class)[int(value) - 1])
    output = np.array(output, dtype=np.float32)
    return output