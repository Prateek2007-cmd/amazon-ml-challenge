for path in ['dataset/test/test_source2.tsv', 'dataset/test/test_source3.tsv']:
    counts = {}
    with open(path, encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 4:
                c = parts[3].strip().upper()
                counts[c] = counts.get(c, 0) + 1
    print(path, counts)
