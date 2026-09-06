import pandas as pd
import matplotlib.pyplot as plt
from config import DATA_DIR, PLOTS_DIR

csv_file_path = DATA_DIR / 'processed' / 'processed_data.csv'
df = pd.DataFrame(pd.read_csv(csv_file_path))

id_counts = df['can_id'].value_counts().sort_values(ascending=True)

plt.figure(figsize=(12, 8))

plt.hlines(y=id_counts.index.astype(str), xmin=0, xmax=id_counts.values,
           color='skyblue', linewidth=3)

plt.plot(id_counts.values, id_counts.index.astype(str), "o",
         markersize=10, color='navy', alpha=0.8)

for i, val in enumerate(id_counts.values):
    formatted_val = f'{val:,}'.replace(',', ' ')
    plt.text(val + 5000, i, formatted_val, va='center', fontsize=10, color='black')

plt.title('CAN ID Frame Frequency', fontsize=16, pad=20)
plt.xlabel('Number of frames', fontsize=14)
plt.ylabel('Module ID (CAN_ID)', fontsize=14)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'ECUfrequency.png')
plt.show()