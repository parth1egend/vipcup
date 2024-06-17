import os
import shutil

def create_directory_structure(base_path):
    """
    Create the directory structure EDSR_Data/0/0, EDSR_Data/1/1, EDSR_Data/2/2.
    """
    for i in range(3):
        dir_path = os.path.join(base_path, 'RCAN_Data', str(i), str(i))
        os.makedirs(dir_path, exist_ok=True)

def move_images(source_dir, base_path):
    """
    Move images from source_dir to the appropriate directories in SR_Data.
    """
    for filename in os.listdir(source_dir):
        if filename[0] in '012':  # Check if the first character is 0, 1, or 2
            src_path = os.path.join(source_dir, filename)
            dest_dir = os.path.join(base_path, 'RCAN_Data', filename[0], filename[0])
            dest_path = os.path.join(dest_dir, filename)
            shutil.copy(src_path, dest_path)

def main():
    source_dir = '/raid/student/2021/ai21btech11005/vipcup/BasicSR/results/RCAN_INFER'  # Replace with your source directory
    base_path = '/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models'  # Replace with the base directory where SR_Data will be created
    
    create_directory_structure(base_path)
    move_images(source_dir, base_path)
    print("Images have been successfully moved.")

if __name__ == "__main__":
    main()
