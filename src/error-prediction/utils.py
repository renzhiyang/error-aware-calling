

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
    complement_mapping = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

    # Complement each nucleotide
    complemented_sequence = ''.join(complement_mapping.get(nuc, nuc) for nuc in reversed_sequence)

    return complemented_sequence