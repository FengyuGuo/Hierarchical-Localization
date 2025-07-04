import argparse
import collections.abc as collections
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

import matplotlib.pyplot as plt

from . import logger
from .utils.io import list_h5_names
from .utils.parsers import parse_image_lists
from .utils.read_write_model import read_images_binary


def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
        if len(names) == 0:
            raise ValueError(f"Could not find any image with the prefix `{prefix}`.")
    elif names is not None:
        if isinstance(names, (str, Path)):
            names = parse_image_lists(names)
        elif isinstance(names, collections.Iterable):
            names = list(names)
        else:
            raise ValueError(
                f"Unknown type of image list: {names}."
                "Provide either a list or a path to a list file."
            )
    else:
        names = names_all
    return names


def get_descriptors(names, path, name2idx=None, key="global_descriptor"):
    if name2idx is None:
        with h5py.File(str(path), "r", libver="latest") as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        desc = []
        for n in names:
            with h5py.File(str(path[name2idx[n]]), "r", libver="latest") as fd:
                desc.append(fd[n][key].__array__())
    return torch.from_numpy(np.stack(desc, 0)).float()


def main(
    descriptors,
    query_prefix=None,
    query_list=None,
    db_prefix=None,
    db_list=None,
    db_model=None,
    db_descriptors=None,
):
    logger.info("Extracting image pairs from a retrieval database.")

    # We handle multiple reference feature files.
    # We only assume that names are unique among them and map names to files.
    if db_descriptors is None:
        db_descriptors = descriptors
    if isinstance(db_descriptors, (Path, str)):
        db_descriptors = [db_descriptors]
    name2db = {n: i for i, p in enumerate(db_descriptors) for n in list_h5_names(p)}
    db_names_h5 = list(name2db.keys())

    query_names_h5 = list_h5_names(descriptors)

    if db_model:
        images = read_images_binary(db_model / "images.bin")
        db_names = [i.name for i in images.values()]
    else:
        db_names = parse_names(db_prefix, db_list, db_names_h5)
    if len(db_names) == 0:
        raise ValueError("Could not find any database image.")
    query_names = parse_names(query_prefix, query_list, query_names_h5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    db_desc = get_descriptors(db_names, db_descriptors, name2db)
    query_desc = get_descriptors(query_names, descriptors)
    sim = torch.einsum("id,jd->ij", query_desc.to(device), db_desc.to(device))

    sim_mat = sim.cpu().numpy()
    plt.imshow(sim_mat, cmap='jet')
    plt.colorbar()
    plt.show()

    exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--query_prefix", type=str, nargs="+")
    parser.add_argument("--query_list", type=Path)
    parser.add_argument("--db_prefix", type=str, nargs="+")
    parser.add_argument("--db_list", type=Path)
    parser.add_argument("--db_model", type=Path)
    parser.add_argument("--db_descriptors", type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
