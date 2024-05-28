# profile_funsearch.py
import sys
from funsearch.__main__ import main
from pathlib import Path
import os

def find_max_idx_in_backups(directory_path):
    """
    This function takes the directory path as input, lists all pickle files containing 'program_db_priority'
    in their names, extracts the idx values from these filenames, and finds the maximum idx value.

    Parameters:
    - directory_path: A string or Path object representing the path to the directory containing the pickle files.

    Returns:
    - max_idx: The maximum idx value found among the filenames. Returns None if no files are found or an error occurs.
    """
    try:
        # Ensure the directory path is a Path object
        pickle_dir = Path(directory_path)

        # List all pickle files in the directory that match the pattern
        pickle_files = list(pickle_dir.glob("program_db_priority_*.pickle"))

        # Extract idx values from each file name and convert them to integers
        idxs = [int(file.stem.split("_")[-2]) for file in pickle_files]

        # Find and return the maximum idx value
        return max(idxs) if idxs else None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
if __name__ == '__main__':
    

    # Simulate command line arguments
    # 1. start a new run
    sys.argv = [
        'funsearch', 
        'run', 
        'examples/worddesign.py', 
        # 'examples/admissibleset.py', 
        # 'examples/shannoncyclic.py', 
        '8', 
        # '--model_name=gpt-4-turbo', 
        # '--model_name=gpt-3.5-turbo-instruct', 
        # '--model_name=nous-hermes-llama2-13b.Q4_0', 
        # '--model_name=wizardlm-13b-v1.2.Q4_0',
        # '--model_name=starcoder-newbpe-q4_0',
        # '--model_name=orca-mini-3b-gguf2-q4_0',
        # '--model_name=codellama-13b-python.Q5_K_M', 
        #'--model_name=codellama-13b-python.Q5_K_S',
        # '--model_name=Meta-Llama-3-8B-Instruct-Q4_K_M', 
        '--model_name=Meta-Llama-3-8B-Instruct-Q5_K_M', 
        #'--model_name=Meta-Llama-3-70B-Instruct.Q4_K_S',
        #'--model_name=codeqwen-1_5-7b-chat-q6_k',
        '--output_path=./data/', 
        '--iterations=-1', 
        '--samplers=1', 
        '--sandbox_type=ExternalProcessSandbox'
    ]

    # idx = find_max_idx_in_backups('./data/backups/')
    # print(f"\nNewest back-up {idx}\n")

    # # # 2. list the results for each island
    # sys.argv = [
    #     'funsearch', 
    #     'ls', 
    #     f'data/backups/program_db_priority_{idx}_0.pickle'  
    # ]

    # 3. restart from a back-up
    # sys.argv = [
    #     'funsearch', 
    #     'run', 
    #     'examples/cap_set_spec.py', 
    #     '5', 
    #     '--model_name=gpt-3.5-turbo-instruct', 
    #     '--output_path=./data/', 
    #     '--iterations=-1', 
    #     '--samplers=15', 
    #     '--sandbox_type=ContainerSandbox', 
    #     f'--load_backup=data/backups/program_db_priority_{idx}_0.pickle'  
    # ]
        
    main()
