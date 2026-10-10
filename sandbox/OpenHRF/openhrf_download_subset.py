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

    # 100 consecutive buffers (50-149), one file per buffer, source opened once
    python openhrf_download_subset.py \
        --source hf://nvidia/OpenH-RF/ulmshare/data/mouse_16_acquisition_1_zea.hdf5 \
        --output "D:/data/OpenH-RF/ulmshare/mouse_16_acquisition_1/buffer_{buffer:03d}.hdf5" \
        --first-frame 20000 --n-frames 400 --n-files 100

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
# Remote reads go through an LRU block cache. HDF5 jumps between the chunk index
# and the chunk data, and fsspec's default read-ahead cache refetches a whole
# block on every such jump; with a block cache the index blocks stay cached.
# 32 MB blocks streamed a 400-frame ULMShare buffer in ~19 s (8 MB: ~34 s).
STREAM_BLOCK_BYTES = 32 * 2**20
STREAM_CACHED_BLOCKS = 16


def _open_source(stack: ExitStack, source: str) -> h5py.File:
    if source.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        path = "datasets/" + source[len("hf://"):]
        f = stack.enter_context(HfFileSystem().open(
            path, "rb", block_size=STREAM_BLOCK_BYTES, cache_type="blockcache",
            cache_options={"maxblocks": STREAM_CACHED_BLOCKS}))
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


def _write_subset(src, raw_path, source, output: Path, first_frame: int, n_frames: int,
                  frames_per_read: int):
    raw = src[raw_path]
    # Write to a temporary name and rename at the end, so an interrupted
    # download never leaves a truncated file that looks complete.
    partial = output.with_name(output.name + ".partial")
    with h5py.File(partial, "w") as dst:
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

    partial.replace(output)
    print(f"Wrote {output} ({output.stat().st_size / 1e6:.0f} MB)", flush=True)


def download_subset(source: str, output: str, first_frame: int, n_frames: int,
                    frames_per_read: int, n_files: int = 1):
    """Copy ``n_files`` consecutive blocks of ``n_frames`` frames, one file each.

    With ``n_files > 1``, ``output`` must contain a ``{buffer}`` placeholder (e.g.
    ``buffer_{buffer:03d}.hdf5``), filled with ``block_first_frame // n_frames``.
    The source is opened once for all blocks (re-opening a remote file re-reads
    its chunk index), and blocks whose output already exists are skipped, so an
    interrupted run can simply be restarted.
    """
    if n_files > 1 and "{buffer" not in output:
        raise ValueError("--output needs a {buffer} placeholder when --n-files > 1")
    with ExitStack() as stack:
        src = _open_source(stack, source)
        raw_path = _raw_data_path(src)
        total = src[raw_path].shape[0]
        print(f"{source}: raw_data {src[raw_path].shape} {src[raw_path].dtype}", flush=True)
        for k in range(n_files):
            block_first = first_frame + k * n_frames
            if block_first < 0 or block_first >= total:
                raise ValueError(f"first frame {block_first} outside [0, {total})")
            block_frames = min(n_frames, total - block_first)
            out_path = Path(output.format(buffer=block_first // n_frames))
            if out_path.exists():
                print(f"Exists, skipping {out_path}", flush=True)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Copying frames {block_first}..{block_first + block_frames - 1} -> {out_path}",
                  flush=True)
            _write_subset(src, raw_path, source, out_path, block_first, block_frames,
                          frames_per_read)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True,
                        help="hf://<repo>/<path> on the Hugging Face Hub, or a local zea .hdf5 file")
    parser.add_argument("--output", required=True,
                        help="Local output .hdf5 file; with --n-files > 1 a template such as "
                             "buffer_{buffer:03d}.hdf5")
    parser.add_argument("--first-frame", type=int, default=0, help="First frame to copy (0-based)")
    parser.add_argument("--n-frames", type=int, default=400, help="Number of frames to copy")
    parser.add_argument("--frames-per-read", type=int, default=40, help="Frames streamed per read")
    parser.add_argument("--n-files", type=int, default=1,
                        help="Number of consecutive --n-frames blocks to copy, one file each")
    args = parser.parse_args()
    download_subset(args.source, args.output, args.first_frame, args.n_frames,
                    args.frames_per_read, args.n_files)


if __name__ == "__main__":
    main()
