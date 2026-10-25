"""Shared utilities for examples."""

import os

import numpy as np
from pyuff_ustb.objects import LinearScan
from pyuff_ustb.objects.uff import Uff

from ustb.tools import download, zenodo_dataset_files_base

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def download_dataset(filename):
    """Download dataset if not already present, return full path."""
    return download(filename, zenodo_dataset_files_base(), DATA_PATH)


def read_channel_data(filename):
    """Download and read channel data from a UFF file."""

    filepath = download_dataset(filename)
    uff_file = Uff(filepath)
    return uff_file.read("channel_data")


def read_uff_object(filename, name):
    """Read any object from a UFF file."""

    filepath = download_dataset(filename)
    uff_file = Uff(filepath)
    return uff_file.read(name)


def make_linear_scan(channel_data, N_x=256, N_z=256, z_max=50e-3):
    """Create a linear scan from channel data probe geometry."""

    probe_x = channel_data.probe.x
    return LinearScan(
        x_axis=np.linspace(float(probe_x.min()), float(probe_x.max()), N_x),
        z_axis=np.linspace(0, z_max, N_z),
    )


def save_image(fig, name):
    """Save figure to examples output directory."""
    outdir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Image saved to {path}")
    return path
