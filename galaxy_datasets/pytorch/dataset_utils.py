import os
import logging
import pandas as pd

from torchvision.transforms import v2
from datasets.distributed import split_dataset_by_node
import datasets as hf_datasets

# utility for adding a validation split to a huggingface dataset dictionary
# can be used inside DataModule.setup(), but best done earlier if doing other operations that require flattening
def add_validation_split(dataset_dict, seed=42, num_workers=4):
    num_workers = max(num_workers, 1)  # at least one worker (pytorch uses 0 to turn offf multiprocessing)
    logging.warning('Creating validation split from 20%% of train dataset, seed ={}'.format(seed))
    train_and_val_dict = dataset_dict["train"].train_test_split(test_size=0.2, shuffle=True, seed=seed, keep_in_memory=seed != 42)
    # now shuffled, so flatten indices
    # breaks (silently hangs) if you have already done set_format
    # https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable#speed-differences
    logging.info('Flattening indices for train and val datasets, may take a while...')
    train_and_val_dict = train_and_val_dict.flatten_indices(num_proc=num_workers, keep_in_memory=seed != 42)
    # don't cache for every random seed, or it will fill up disk space
    dataset_dict['train'] = train_and_val_dict["train"]
    dataset_dict['validation'] = train_and_val_dict["test"]
    del train_and_val_dict
    return dataset_dict


def distribute_dataset_with_lightning(dataset_dict: hf_datasets.DatasetDict):
    # split dataset for each rank, using slurm env variables

    # with lightning, these aren't set as you'd expect, remain unset
    # rank = int(os.environ.get("LOCAL_RANK", 0))  # local rank for this process
    # world_size = int(os.environ.get("WORLD_SIZE", 1))  # total number of processes

    # use env variables from slurm instead
    # distribute_dataset_with_lightning()
    rank = int(os.environ.get("SLURM_PROCID", 0))  # index of slurm task
    world_size = int(os.environ.get("SLURM_NTASKS", 1))  # total number of slurm tasks

    # requested_nodes = int(os.environ['SLURM_NNODES'])  # e.g. 4
    # requested_tasks_per_node = int(os.environ['SLURM_TASKS_PER_NODE'].split('(')[0])  # e.g. 2(x4) -> 2


    logging.info('Beginning data loading on rank {}, world size {}'.format(rank, world_size))

    if world_size > 1:
        logging.info(f"Distributing datasets on rank {rank}, world size {world_size}")
        for split in dataset_dict.keys():
            if split != 'test':  # never distribute test set to ensure all rows evaluated exactly once
                logging.info(f"Selecting from {split}")
                dataset_dict[split]  = split_dataset_by_node(dataset_dict[split], rank=rank, world_size=world_size)
    else:
        logging.info(f"Not distributing datasets on rank {rank}, world {world_size}, single gpu training")

    return dataset_dict

# pre-decode from PIL to tensor to save cpu at the cost of I/O
# TODO change default transform to use toImage for simplicity?
def pil_to_tensors(dataset: hf_datasets.Dataset, num_workers=1):
    transform_to_tensor = v2.PILToTensor()  # no compose needed
    def transform_to_tensor_wrapped(example):
        example['image'] = transform_to_tensor(example['image'])
        return example
    return dataset.map(transform_to_tensor_wrapped, num_proc=num_workers)
