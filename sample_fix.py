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

    # 1. Safe Load & Repair (Fixes the ParserError)
    cleaned_rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # If header has 11 columns, remove 'hand_to_ear' (index 5)
        if len(header) == 11:
            header.pop(5)
            
        for row in reader:
            if not row: continue # Skip empty lines
            
            # If row has 11 columns, remove the 6th item to match new 10-col format
            if len(row) == 11:
                row.pop(5)
            
            # Only keep rows that now match our 10-column header
            if len(row) == len(header):
                cleaned_rows.append(row)

    # Convert the repaired list into a DataFrame
    df = pd.DataFrame(cleaned_rows, columns=header)
    df = df.apply(pd.to_numeric, errors='ignore') # Convert numeric strings to floats/ints
    df = df.dropna()
    
    original_count = len(df)
    print(f"Successfully loaded and repaired {original_count} samples.")

    # 2. Remove exact duplicates
    df = df.drop_duplicates()
    
    # 3. Near-Duplicate Filtering (Distance-based)
    features = df.drop('label', axis=1).values
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
                    
    df_filtered = df.drop(df.index[list(to_exclude)])
    
    # 4. Shuffle the dataset
    df_shuffled = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_count = len(df_shuffled)
    
    # 5. Overwrite the original file with the clean 10-column data
    df_shuffled.to_csv(path, index=False)
    
    print(f"Processing complete.")
    print(f"Samples removed (duplicates/similar): {original_count - final_count}")
    print(f"Final dataset size: {final_count} columns: {len(df_shuffled.columns)}")
    print(f"Saved to: {path}")

# Execute
repair_and_process_dataset(file_path)