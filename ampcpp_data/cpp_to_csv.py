import sys
 
seqs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
 
with open(sys.argv[2], "w") as f:
    f.write("id,sequence,label\n")
    for i, s in enumerate(seqs, 1):
        f.write(f"cpp{i},{s},0\n")
 
print(len(seqs), "rows")
