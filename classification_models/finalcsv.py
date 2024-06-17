import os
import pandas as pd

# Directory containing the CSV files
folder_path = '/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/predictions_ESRGAN'

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Add "RawData" in front of the image names
        df['image_name'] = df['image_name'].apply(lambda x: f'RawData{x}')
        
        # Save the modified DataFrame back to a CSV file
        df.to_csv(file_path, index=False)

print("All files have been updated.")
