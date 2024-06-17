import os
import pandas as pd

# Define the folder containing the text files
folder_path = '/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/testing'

# Initialize lists to store the model names, test loss, and test accuracy
models = []
test_losses = []
test_accuracies = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        with open(os.path.join(folder_path, filename), 'r') as file:
            data = file.readlines()
            model_name = filename.replace('.txt', '')
            test_loss = float(data[0].split(':')[1].strip())
            test_accuracy = float(data[1].split(':')[1].strip().replace('%', ''))
            
            models.append(model_name)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

# Create a DataFrame from the collected data
df = pd.DataFrame({
    'Model': models,
    'Test Loss': test_losses,
    'Test Accuracy (%)': test_accuracies
})

# Save the DataFrame to a CSV file
csv_path = '/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/model_performance_denoisedEDSR_patient_split.csv'
df.to_csv(csv_path, index=False)

# Convert the CSV file to an XLSX file
xlsx_path = '/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/model_performance_denoisedEDSR_patient_split.xlsx'
df.to_excel(xlsx_path, index=False)

print(f'Data successfully written to {xlsx_path}')
