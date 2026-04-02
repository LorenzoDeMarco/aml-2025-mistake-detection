import numpy as np
import sys
import os

def inspect_npz(file_path):
    """
    Load an .npz file and print the shape and data type of its arrays.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    try:
        # Load the .npz file
        data = np.load(file_path)
        
        print(f"Inspecting: {file_path}")
        print("-" * 40)
        
        # Iterate over all arrays stored in the .npz file
        for key in data.files:
            array = data[key]
            print(f"Key   : {key}")
            print(f"Shape : {array.shape}")
            print(f"Type  : {array.dtype}")
            print("-" * 40)
            
        data.close()
    except Exception as e:
        print(f"Failed to read the .npz file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        default_path = "feature_extraction/1_7_224_360p.mp4_1.875hz.npz"
        print(f"Trying default path: {default_path}\n")
        inspect_npz(default_path)
    else:
        file_path = sys.argv[1]
        inspect_npz(file_path)