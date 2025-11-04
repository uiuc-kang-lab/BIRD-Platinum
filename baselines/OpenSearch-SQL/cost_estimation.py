n_input_tokens = 0
n_output_tokens = 0
n_processed_data = 0
with open("slurm-5478599.out") as f:
    for line in f:
        if "Current input tokens:" in line:
            n_input_tokens += int(line.split()[-1])
        if "Current output tokens:" in line:
            n_output_tokens += int(line.split()[-1])
        if "/498" in line:
            n_processed_data = int(line.split()[-1].split("/")[0])

cost = 2.5 * n_input_tokens / 1000000 + 10.0 * n_output_tokens / 1000000
print(f"Processed data: {n_processed_data}")
print(f"Input tokens: {n_input_tokens}")
print(f"Output tokens: {n_output_tokens}")
print(f"Estimated cost per question: ${cost/n_processed_data:.4f}")