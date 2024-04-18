import pandas as pd
import matplotlib.pyplot as plt

file_path = "simulations/2024-04-18 1718/output.csv"

# Read the CSV file
df = pd.read_csv(file_path)

score_pos = abs(df["score"][0])

# Group data into batches of 1000 epochs
batch_size = 10
num_batches = (len(df) + batch_size - 1) // batch_size
batches = [df[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
# Initialize lists to store counts
blue_counts = []
red_counts = []
cumulative_blue = 0
cumulative_red = 0

# Iterate over batches
for batch in batches:
    # Count scores equal to -5 and other scores
    blue_count = len(batch[batch['score'] == score_pos])
    red_count = len(batch) - blue_count
    
    # Update cumulative counts
    cumulative_blue += blue_count
    cumulative_red += red_count
    
    # Append counts to lists
    blue_counts.append(blue_count)
    red_counts.append(red_count)

# Create cumulative bar plot
plt.figure(figsize=(10, 6))
plt.bar(range(num_batches), red_counts, color='red', 
        label=f"Score = -{score_pos}")
plt.bar(range(num_batches), blue_counts, color='blue', 
        label=f"Score = {score_pos}", bottom=red_counts)
plt.xlabel('Batch Number')
plt.ylabel('Cumulative Count')
plt.title('Cumulative Bar Plot of Scores per Batch of 10 simulations')
plt.legend()

# Show plot
plt.show()
