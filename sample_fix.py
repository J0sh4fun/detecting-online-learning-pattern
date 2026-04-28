import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import csv

# Path to your dataset
file_path = 'data/posture_dataset.csv'

def repair_and_process_dataset(path, threshold=0.005):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return

    # 1. Safe Load & Repair
    cleaned_rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        if len(header) == 11:
            header.pop(5)
            
        for row in reader:
            if not row: continue 
            if len(row) == 11:
                row.pop(5)
            if len(row) == len(header):
                cleaned_rows.append(row)

    df = pd.DataFrame(cleaned_rows, columns=header)
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.dropna()
    
    original_count = len(df)
    print(f"Successfully loaded {original_count} samples.")

    # --- NEW LOGIC: Isolate Absence labels to prevent removal ---
    df_absence = df[df['label'] == 'Absence']
    df_others = df[df['label'] != 'Absence']

    # 2. Remove exact duplicates ONLY from non-absence data
    df_others = df_others.drop_duplicates()
    
    # 3. Near-Duplicate Filtering ONLY from non-absence data
    if not df_others.empty:
        features = df_others.drop('label', axis=1).values
        nn = NearestNeighbors(radius=threshold)
        nn.fit(features)
        
        adj_matrix = nn.radius_neighbors_graph(features).toarray()
        
        to_exclude = set()
        for i in range(len(adj_matrix)):
            if i not in to_exclude:
                neighbors = np.where(adj_matrix[i] == 1)[0]
                for n in neighbors:
                    if n != i:
                        to_exclude.add(n)
                        
        df_filtered_others = df_others.drop(df_others.index[list(to_exclude)])
    else:
        df_filtered_others = df_others

    # 4. Re-combine the filtered data with the original Absence labels
    df_final = pd.concat([df_absence, df_filtered_others])
    
    # 5. Shuffle the entire dataset
    df_shuffled = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_count = len(df_shuffled)
    
    # 6. Overwrite the original file
    df_shuffled.to_csv(path, index=False)
    
    print(f"Processing complete.")
    print(f"Samples removed: {original_count - final_count}")
    print(f"Absence samples preserved: {len(df_absence)}")
    print(f"Final dataset size: {final_count}")
    print(f"Saved to: {path}")

# Execute
repair_and_process_dataset(file_path)