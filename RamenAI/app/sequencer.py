
def sequence_line(line, n):
    sequences = list()
    for i in range(n, len(line)) :
        sequence = line[i-n: i]
        sequences.append(sequence)
    return sequences

