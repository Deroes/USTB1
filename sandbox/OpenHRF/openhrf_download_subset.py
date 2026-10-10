"""Stream a frame subset of an OpenH-RF (zea) HDF5 file to a local HDF5 file.

OpenH-RF acquisitions are large (the ULMShare files are 60-370 GB each) and the
raw data is Blosc-compressed, which MATLAB cannot read without an HDF5 filter
plugin. This script streams only the requested frames from the Hugging Face Hub
(or from a local zea file) and writes them to a small local file with the
**same zea layout**, recompressed with gzip so that MATLAB's ``h5read`` can read
it directly. The local file is then converted to USTB with
``openhrf_to_uff.m``.

Everything except ``raw_data`` (probe, scan, metadata, custom, root attributes)
is copied verbatim. The selected frame range is recorded in the root attributes
``openhrf_source``, ``openhrf_first_frame`` and ``openhrf_n_frames``.

Usage::

    # one 400-frame ULMShare buffer of mouse 10, acquisition 1 (~720 MB local)
    python openhrf_download_subset.py \
        --source hf://nvidia/OpenH-RF/ulmshare/data/mouse_10_acquisition_1_zea.hdf5 \
        --output C:/Data/OpenH-RF/ulmshare/mouse_10_acquisition_1_frames_20000_20399.hdf5 \
        --first-frame 20000 --n-frames 400

Requirements::

    pip install h5py hdf5plugin huggingface_hub numpy
"""

from __future__ import annotations

import argparse
import time
from contextlib import ExitStack
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the Blosc filter used by OpenH-RF raw_data)
import numpy as np

RAW_DATA_KEY = "data/raw_data"
STREAM_BLOCK_BYTES = 64 * 2**20  # fewer, larger HTTP range requests (HF rate limit)


def _open_source(stack: ExitStack, source: str) -> h5py.File:
    if source.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        path = "datasets/" + source[len("hf://"):]
        f = stack.enter_context(HfFileSystem().open(path, "rb", block_size=STREAM_BLOCK_BYTES))
        return stack.enter_context(h5py.File(f, "r"))
    return stack.enter_context(h5py.File(source, "r"))


def _copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _copy_tree(src: h5py.Group, dst: h5py.Group, skip: str):
    """Copy every group/dataset under ``src`` except the dataset at path ``skip``."""
    _copy_attrs(src, dst)
    for name, obj in src.items():
        if obj.name == skip:
            continue
        if isinstance(obj, h5py.Group):
            _copy_tree(obj, dst.require_group(name), skip)
        else:
            dset = dst.create_dataset(name, data=obj[()])
            _copy_attrs(obj, dset)


def _raw_data_path(h5: h5py.File) -> str:
    """zea >= 0.1.4 stores data under /tracks/track_0; older files at the root."""
    for candidate in ("/tracks/track_0/" + RAW_DATA_KEY, "/" + RAW_DATA_KEY):
        if candidate in h5:
            return candidate
    raise KeyError(f"No {RAW_DATA_KEY} in file (looked in /tracks/track_0 and /)")


def download_subset(source: str, output: Path, first_frame: int, n_frames: int, frames_per_read: int):
    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        src = _open_source(stack, source)
        raw_path = _raw_data_path(src)
        raw = src[raw_path]
        total = raw.shape[0]
        if first_frame < 0 or first_frame >= total:
            raise ValueError(f"--first-frame {first_frame} outside [0, {total})")
        n_frames = min(n_frames, total - first_frame)
        print(f"{source}\n  raw_data {raw.shape} {raw.dtype}; copying frames "
              f"{first_frame}..{first_frame + n_frames - 1}")

        with h5py.File(output, "w") as dst:
            _copy_tree(src, dst, skip=raw_path)
            dst.attrs["openhrf_source"] = source
            dst.attrs["openhrf_first_frame"] = np.int64(first_frame)
            dst.attrs["openhrf_n_frames"] = np.int64(n_frames)

            out = dst.create_dataset(
                raw_path,
                shape=(n_frames,) + raw.shape[1:],
                dtype=raw.dtype,
                chunks=(1,) + raw.shape[1:],
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            _copy_attrs(raw, out)

            t_start = time.time()
            for i0 in range(0, n_frames, frames_per_read):
                i1 = min(i0 + frames_per_read, n_frames)
                out[i0:i1] = raw[first_frame + i0:first_frame + i1]
                elapsed = time.time() - t_start
                print(f"  frames {i1}/{n_frames}  ({elapsed:.0f} s)", flush=True)

    print(f"Wrote {output} ({output.stat().st_size / 1e6:.0f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True,
                        help="hf://<repo>/<path> on the Hugging Face Hub, or a local zea .hdf5 file")
    parser.add_argument("--output", required=True, type=Path, help="Local output .hdf5 file")
    parser.add_argument("--first-frame", type=int, default=0, help="First frame to copy (0-based)")
    parser.add_argument("--n-frames", type=int, default=400, help="Number of frames to copy")
    parser.add_argument("--frames-per-read", type=int, default=40, help="Frames streamed per read")
    args = parser.parse_args()
    download_subset(args.source, args.output, args.first_frame, args.n_frames, args.frames_per_read)


if __name__ == "__main__":
    main()
