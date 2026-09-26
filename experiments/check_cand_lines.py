with open('output/candidate_pairs.tsv', encoding='utf-8') as f:
    f.readline()
    for i in range(1, 2010):
        line = f.readline()
        if i in (1, 2, 1965, 1966, 1967, 1968, 2000):
            print(f"Line {i}: {line[:60]}... (len={len(line.strip().split(chr(9)))})")
