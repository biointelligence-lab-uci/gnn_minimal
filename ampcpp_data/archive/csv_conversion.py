import csv

cpp_file = "Poreforming_CPPs.txt"
amp_file = "Poreforming_AMPS.txt"
output_file = "AMPCPP_dataset.csv"

datasets = [
    (cpp_file, 0),  # Label 0 for CPPs
    (amp_file, 1),  # Label 1 for AMPs
]

current_id = 1  # Start at 1 for seq1, seq2... (use 0 if you prefer seq0, seq1...)

with open(output_file, "w", newline="") as csv_out:
  writer = csv.writer(csv_out)
  # Write the header row
  writer.writerow(["id", "label", "sequence"])

  for filepath, label in datasets:
    with open(filepath, "r") as txt_in:
      for line in txt_in:
        sequence = line.strip()
        if sequence:
          # Format the ID as 'seq1', 'seq2', etc.
          formatted_id = f"seq{current_id}"
          writer.writerow([formatted_id, label, sequence])
          current_id += 1

print(f"Dataset created with {current_id - 1} sequences!")