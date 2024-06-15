import filecmp
import os
import sys
import tempfile
import time

from pathlib import Path

import compress

file_folder = Path(__file__).parent / "files"

if not os.path.isdir(file_folder):
    print("Error: files folder not found")
    sys.exit(1)


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


total_comp_time, total_decomp_time = 0, 0
total_comp_size, total_decomp_size = 0, 0


def test_file_compression(temp_dir: str, file: Path):
    print(f"Testing file: {file.name} ({sizeof_fmt(file.stat().st_size)})")
    temp_file_path = temp_dir + "/" + file.name

    uncompressed_size = file.stat().st_size
    start_time = time.perf_counter()
    compress.compress_file(str(file), temp_file_path + ".huf")
    stop_time = time.perf_counter()
    compress_time = stop_time - start_time
    comp_throughput = f"{sizeof_fmt(uncompressed_size / compress_time)}/s"
    print(f"Compressed in {compress_time:.4f} seconds. ({comp_throughput})")

    compressed_size = Path(temp_file_path + ".huf").stat().st_size
    start_time = time.perf_counter()
    compress.decompress_file(temp_file_path + ".huf", temp_file_path)
    stop_time = time.perf_counter()
    decompress_time = stop_time - start_time
    comp_throughput = f"{sizeof_fmt(compressed_size / decompress_time)}/s"
    print(f"Decompressed in {decompress_time:.4f} seconds. ({comp_throughput})")

    comp_ratio = uncompressed_size / compressed_size
    comp_percent = compressed_size / uncompressed_size * 100
    print(f"Compression size: {sizeof_fmt(compressed_size)}")
    print(f"Compression ratio: {comp_ratio:.4f} ({comp_percent:.4f}%)")

    global total_comp_time, total_decomp_time
    global total_decomp_size, total_comp_size
    total_comp_time += compress_time
    total_decomp_time += decompress_time
    total_comp_size += compressed_size
    total_decomp_size += uncompressed_size

    if not filecmp.cmp(file, temp_dir + "/" + file.name):
        raise RuntimeError(
            f"File {file.name} was not compressed or decompressed correctly."
        )


with tempfile.TemporaryDirectory() as temp_dir:
    print(f"Created temporary directory: {temp_dir}")

    file_sizes = sorted(
        (file.stat().st_size, file)
        for file in file_folder.iterdir()
        if file.is_file()
    )
    files = [file for _, file in file_sizes]

    print("Testing files in order of size:\n")
    for file in files:
        test_file_compression(temp_dir, file)
        print()

    comp_throughput = sizeof_fmt(total_decomp_size / total_comp_time)
    decomp_throughput = sizeof_fmt(total_comp_size / total_decomp_time)
    print(
        f"Total: {len(files)} files "
        f"(Original: {sizeof_fmt(total_decomp_size)}) "
        f"(Compressed: {sizeof_fmt(total_comp_size)})\n"
        f"Total compression time: {total_comp_time:.4f} seconds. "
        f"({comp_throughput}/s)\n"
        f"Total decompression time: {total_decomp_time:.4f} seconds. "
        f"({decomp_throughput}/s)\n"
        f"Total time: {total_comp_time + total_decomp_time:.4f} seconds."
    )
