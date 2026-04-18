# generate_multi_source_data.py
# Generate train/val/test subsets in one run.
# 0° points to mic-0 (left), 180° points to mic-3 (right).

import argparse
import csv
import multiprocessing as mp
import os
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning

warnings.filterwarnings("ignore", category=WavFileWarning)

# ========== Scene / microphone settings (pickle-safe module constants) ==========
ROOM_DIM = np.array([8.0, 8.0, 5.0])
MIC_CENTER = np.array([4.0, 4.0, 2.5])
NUM_MIC = 4
DIAMETER = 0.08
SPACING = DIAMETER / (NUM_MIC - 1)
FS = 16000
SIG_DUR = 1.0
DIST_RANGE = (0.5, 2.5)
RT60_RANGE = (0.2, 1.2)
SNR_RANGE = (0, 30)

offsets = np.linspace(-SPACING * (NUM_MIC - 1) / 2, SPACING * (NUM_MIC - 1) / 2, NUM_MIC)
MIC_COORDS = np.stack(
    [
        MIC_CENTER[0] + offsets,
        np.full(NUM_MIC, MIC_CENTER[1]),
        np.full(NUM_MIC, MIC_CENTER[2]),
    ]
)

ALL_AZ = np.arange(0, 181, 1)
EL_FIXED = 0

# Worker pool: source list (set once per process via initializer)
_POOL_SRC_LIST: list[Path] | None = None


def _pool_init(src_strs: list[str]) -> None:
    global _POOL_SRC_LIST
    _POOL_SRC_LIST = [Path(s) for s in src_strs]


def load_audio(path: Path, target_len: int = 16000):
    """Load single-channel normalized audio and pad/crop to target length."""
    try:
        sr, sig = wavfile.read(path)
    except Exception as e:
        print(f"\nWarning: failed to read {path.name}: {e}. Skipping.")
        return None

    if sr != FS:
        try:
            import resampy

            if sig.ndim > 1:
                sig = sig[:, 0]
            sig = resampy.resample(sig, sr, FS)
        except ImportError:
            raise ImportError("Please install `resampy` (pip install resampy).")

    if sig.dtype != np.float32:
        if sig.ndim > 1:
            sig = sig[:, 0]
        if np.max(np.abs(sig)) == 0:
            print(f"\nWarning: {path.name} is silent. Skipping.")
            return None
        max_val = np.iinfo(sig.dtype).max or float(np.max(np.abs(sig)))
        sig = sig.astype(np.float32) / max_val

    if len(sig) < target_len:
        sig = np.pad(sig, (0, target_len - len(sig)), "wrap")
    else:
        sig = sig[:target_len]
    return sig


def user_az_to_direction(az_deg):
    """Convert user angle convention to 3D direction vector."""
    math_az_rad = np.deg2rad(180 - az_deg)
    el_rad = np.deg2rad(EL_FIXED)
    return np.array(
        [
            np.cos(el_rad) * np.cos(math_az_rad),
            np.cos(el_rad) * np.sin(math_az_rad),
            np.sin(el_rad),
        ]
    )


def generate_one_sample(num_src: int, src_list: list[Path]):
    """Simulate one sample and return (wav_data, az_list)."""
    rt60 = np.random.uniform(*RT60_RANGE)
    e_abs, max_order = pra.inverse_sabine(rt60, ROOM_DIM.tolist())
    room = pra.ShoeBox(
        ROOM_DIM,
        fs=FS,
        materials=pra.Material(e_abs),
        max_order=max_order,
        air_absorption=True,
    )
    room.add_microphone_array(pra.MicrophoneArray(MIC_COORDS, FS))

    az_list = []

    if num_src > 0:
        chosen_az = sorted(random.sample(list(ALL_AZ), num_src))
        for az_deg in chosen_az:
            sig = None
            while sig is None:
                src_path = random.choice(src_list)
                sig = load_audio(src_path, target_len=int(SIG_DUR * FS))

            dist = np.random.uniform(*DIST_RANGE)
            direction = user_az_to_direction(az_deg)
            src_pos = MIC_CENTER + dist * direction
            src_pos = np.clip(src_pos, 0.02, ROOM_DIM - 0.02)
            room.add_source(src_pos, signal=sig)
            az_list.append(az_deg)

    if num_src == 0:
        target_len = int(SIG_DUR * FS)
        noise_level = np.random.uniform(5e-4, 5e-3)
        noise = np.random.randn(target_len, NUM_MIC).astype(np.float32) * noise_level
        wav_data = (noise * 32767).astype(np.int16)
    else:
        room.simulate(snr=np.random.uniform(*SNR_RANGE))
        wav_data = (room.mic_array.signals.T * 32767).astype(np.int16)

    return wav_data, az_list


def _write_one(uid: str, split_wav: Path, split_lbl: Path, num_src: int, src_list: list[Path]) -> None:
    wav_data, az_list = generate_one_sample(num_src, src_list)
    wavfile.write(split_wav / f"{uid}.wav", FS, wav_data)
    with open(split_lbl / f"{uid}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([num_src])
        for az in az_list:
            az_out = Decimal(str(az)).quantize(Decimal("0.0"), ROUND_HALF_UP)
            writer.writerow([az_out])


def _worker_write_one(task: tuple[str, str, str, int]) -> None:
    """Pickle-friendly worker: (split_wav_dir, split_lbl_dir, uid, num_src)."""
    sw, sl, uid, num_src = task
    assert _POOL_SRC_LIST is not None
    _write_one(uid, Path(sw), Path(sl), num_src, _POOL_SRC_LIST)


def _default_workers() -> int:
    return min(os.cpu_count() or 1, 16)


def _executor_mp_context():
    """Prefer fork on Linux/macOS to avoid re-import side effects; Windows uses spawn."""
    if sys.platform == "win32":
        return None
    try:
        return mp.get_context("fork")
    except ValueError:
        return None


def generate_split(
    split_name: str,
    split_samples: int,
    base: str,
    split_wav: Path,
    split_lbl: Path,
    src_list: list[Path],
    workers: int,
) -> None:
    """Generate one split with balanced source counts 0/1/2/3."""
    if split_samples == 0:
        print(f"\nSkipping split '{split_name}' (0 samples).")
        return

    samples_per_class = split_samples // 4
    print(f"\n=== Generating split '{split_name}' ({split_samples} samples, {samples_per_class} per class) ===")

    local_idx = 0
    tasks: list[tuple[str, str, str, int]] = []
    for num_src in range(4):
        for j in range(1, samples_per_class + 1):
            local_idx += 1
            uid = f"{base}-{split_name}-{local_idx:05d}"
            tasks.append((str(split_wav), str(split_lbl), uid, num_src))

    sw_s, sl_s = str(split_wav), str(split_lbl)
    n_tasks = len(tasks)

    if workers <= 1:
        for i, (_, _, uid, num_src) in enumerate(tasks):
            _write_one(uid, Path(sw_s), Path(sl_s), num_src, src_list)
            if (i + 1) % 500 == 0 or (i + 1) == n_tasks:
                print(f"     {i + 1}/{n_tasks} done for split '{split_name}'")
        return

    src_strs = [str(p) for p in src_list]
    chunksize = max(1, n_tasks // (workers * 8))
    ctx = _executor_mp_context()
    print(f"  Using {workers} parallel workers (chunksize={chunksize})...")

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_pool_init,
        initargs=(src_strs,),
    ) as ex:
        for i, _ in enumerate(ex.map(_worker_write_one, tasks, chunksize=chunksize)):
            if (i + 1) % 500 == 0 or (i + 1) == n_tasks:
                print(f"     {i + 1}/{n_tasks} done for split '{split_name}'")


def main() -> None:
    root_out = Path(__file__).resolve().parent
    src_root = root_out / "original_dataset"

    parser = argparse.ArgumentParser(
        description="Generate multi-source room data. Without --name, runs interactively."
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Dataset folder name and filename prefix (e.g. 10k). Omit for interactive mode.",
    )
    parser.add_argument(
        "--train",
        type=int,
        default=None,
        help="Number of TRAIN samples (positive multiple of 4). Required with --name.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Val and test size relative to train each (default 0.25). Optional with --name.",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Parallel worker processes (1 = serial). Default: min(CPU count, 16).",
    )
    args = parser.parse_args()

    print(f"Scanning source files from '{src_root}' and subfolders...")
    src_list = list(src_root.rglob("*.wav"))
    if not src_list:
        src_list.extend(list(src_root.rglob("*.WAV")))
    assert len(src_list) >= 1, f"No .wav files found under '{src_root}'."
    print(f"Found {len(src_list)} source files.")

    if args.name is not None:
        base = args.name.strip() or "multi"
        if args.train is None:
            parser.error("--train is required when --name is set")
        train_samples = args.train
        val_test_ratio = args.ratio if args.ratio is not None else 0.25
    else:
        base = input("Enter dataset name (used for output folder and filename prefix, e.g. multi): ").strip()
        if not base:
            base = "multi"
            print(f"Empty name detected, using default: {base}")

        train_input = input(
            "Enter number of TRAIN samples (must be a positive multiple of 4, e.g. 5000): "
        ).strip()
        train_samples = int(train_input)
        ratio_input = input(
            "Enter validation/test ratio relative to train (default 0.25 means val=train*0.25 and test=train*0.25): "
        ).strip()
        val_test_ratio = float(ratio_input) if ratio_input else 0.25

    assert train_samples > 0 and train_samples % 4 == 0, (
        "TRAIN samples must be a positive integer divisible by 4."
    )
    assert val_test_ratio >= 0, "Validation/test ratio must be >= 0."

    val_samples = int(round(train_samples * val_test_ratio))
    test_samples = val_samples
    val_samples = (val_samples // 4) * 4
    test_samples = (test_samples // 4) * 4

    total_samples = train_samples + val_samples + test_samples
    print("\nPlanned split:")
    print(f"  train: {train_samples}")
    print(f"  val:   {val_samples}")
    print(f"  test:  {test_samples}")
    print(f"  total: {total_samples}")

    dataset_root = root_out / "../datacreate/dataset" / base
    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    split_dirs: dict[str, tuple[Path, Path]] = {}
    for split_name in splits.keys():
        sw = dataset_root / split_name / "wav"
        sl = dataset_root / split_name / "label"
        sw.mkdir(parents=True, exist_ok=True)
        sl.mkdir(parents=True, exist_ok=True)
        split_dirs[split_name] = (sw, sl)

    workers = args.workers if args.workers is not None else _default_workers()
    workers = max(1, workers)
    if workers > 1 and sys.platform == "win32":
        print("Note: Windows uses 'spawn'; first pool startup may be slower.")

    print("\nStarting data generation...")
    t0 = time.perf_counter()

    for split_name, split_samples in splits.items():
        sw, sl = split_dirs[split_name]
        generate_split(split_name, split_samples, base, sw, sl, src_list, workers)

    t1 = time.perf_counter()
    print(f"\nDone. Dataset root: {dataset_root}")
    print(f"Total generated samples: {total_samples}")
    print(f"Elapsed time: {t1 - t0:.1f} seconds")


if __name__ == "__main__":
    main()
