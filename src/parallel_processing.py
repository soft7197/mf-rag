import os
import torch
import multiprocessing
import subprocess

def get_available_gpus():
    """ Returns a list of available GPU device IDs. """
    return [i for i in range(torch.cuda.device_count())]

def process_file_with_gpu(file, gpu_id):
    """ Assigns a bug processing task to a specific GPU. """
    file_path = os.path.join("/home/selab/Desktop/dlt-conf/java/selab/extracted_methods", file)
    print(f"Processing {file} on GPU {gpu_id}...")
    subprocess.run(["python3", "embed_and_index.py", file_path, str(gpu_id)])

def main():
    dir_path = "/home/selab/Desktop/dlt-conf/java/selab/extracted_methods"
    files = os.listdir(dir_path)

    gpus = get_available_gpus()
    num_gpus = len(gpus)

    print(f"Using {num_gpus} GPUs: {gpus}")

    # Create multiprocessing pool with GPU assignment
    with multiprocessing.Pool(processes=num_gpus) as pool:
        pool.starmap(process_file_with_gpu, [(file, i % num_gpus) for i, file in enumerate(files)])

if __name__ == "__main__":
    main()
