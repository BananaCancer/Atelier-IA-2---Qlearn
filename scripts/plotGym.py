import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
file_path = "simulations/2024-04-18 1757/output.csv"

# Read the CSV file
df = pd.read_csv(file_path)

score_pos = abs(df["score"][0])

# Create cumulative bar plot
plt.figure(figsize=(10, 6))
plt.plot(df["step_num"])
plt.xlabel("Numéro de la simulation")
plt.ylabel('Nombre d\'itérations ')
plt.title("Graphique représentant le nombre d'itérations pour finir le jeu IceWalking")

# Show plot
plt.show()
