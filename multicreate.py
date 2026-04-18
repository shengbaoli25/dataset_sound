# generate_multi_source_data.py
# Generate train/val/test subsets in one run.
# 0° points to mic-0 (left), 180° points to mic-3 (right).

import time
from pathlib import Path
import numpy as np, csv, random
import pyroomacoustics as pra
from scipy.io import wavfile
from decimal import Decimal, ROUND_HALF_UP
import warnings
from scipy.io.wavfile import WavFileWarning

warnings.filterwarnings("ignore", category=WavFileWarning)

# ========== Source audio root ==========
ROOT_OUT = Path.cwd()
SRC_ROOT = ROOT_OUT / "original_dataset"

print(f"Scanning source files from '{SRC_ROOT}' and subfolders...")
SRC_LIST = list(SRC_ROOT.rglob("*.wav"))
if not SRC_LIST:
    SRC_LIST.extend(list(SRC_ROOT.rglob("*.WAV")))
assert len(SRC_LIST) >= 1, f"No .wav files found under '{SRC_ROOT}'."
print(f"Found {len(SRC_LIST)} source files.")

# ========== User input ==========
BASE = input("Enter dataset name (used for output folder and filename prefix, e.g. multi): ").strip()
if not BASE:
    BASE = "multi"
    print(f"Empty name detected, using default: {BASE}")

train_input = input("Enter number of TRAIN samples (must be a positive multiple of 4, e.g. 5000): ").strip()
TRAIN_SAMPLES = int(train_input)
assert TRAIN_SAMPLES > 0 and TRAIN_SAMPLES % 4 == 0, "TRAIN samples must be a positive integer divisible by 4."

ratio_input = input(
    "Enter validation/test ratio relative to train (default 0.25 means val=train*0.25 and test=train*0.25): "
).strip()
VAL_TEST_RATIO = float(ratio_input) if ratio_input else 0.25
assert VAL_TEST_RATIO >= 0, "Validation/test ratio must be >= 0."

VAL_SAMPLES = int(round(TRAIN_SAMPLES * VAL_TEST_RATIO))
TEST_SAMPLES = VAL_SAMPLES

# Make each split divisible by 4 to keep class balance for 0/1/2/3 sources.
VAL_SAMPLES = (VAL_SAMPLES // 4) * 4
TEST_SAMPLES = (TEST_SAMPLES // 4) * 4

TOTAL_SAMPLES = TRAIN_SAMPLES + VAL_SAMPLES + TEST_SAMPLES
print("\nPlanned split:")
print(f"  train: {TRAIN_SAMPLES}")
print(f"  val:   {VAL_SAMPLES}")
print(f"  test:  {TEST_SAMPLES}")
print(f"  total: {TOTAL_SAMPLES}")

# ========== Output directories ==========
DATASET_ROOT = ROOT_OUT / "../datacreate/dataset" / BASE
SPLITS = {
    "train": TRAIN_SAMPLES,
    "val": VAL_SAMPLES,
    "test": TEST_SAMPLES,
}

SPLIT_DIRS = {}
for split_name in SPLITS.keys():
    split_wav = DATASET_ROOT / split_name / "wav"
    split_lbl = DATASET_ROOT / split_name / "label"
    split_wav.mkdir(parents=True, exist_ok=True)
    split_lbl.mkdir(parents=True, exist_ok=True)
    SPLIT_DIRS[split_name] = (split_wav, split_lbl)

# ========== Scene / microphone settings ==========
ROOM_DIM = np.array([8., 8., 5.])
MIC_CENTER = np.array([4., 4., 2.5])
NUM_MIC = 4
DIAMETER = 0.08
SPACING = DIAMETER / (NUM_MIC - 1)
FS = 16000
SIG_DUR = 1.0
DIST_RANGE = (0.5, 2.5)
RT60_RANGE = (0.2, 1.2)
SNR_RANGE = (0, 30)

# ========== Build linear array ==========
offsets = np.linspace(-SPACING * (NUM_MIC - 1) / 2,
                      SPACING * (NUM_MIC - 1) / 2, NUM_MIC)
MIC_COORDS = np.stack([
    MIC_CENTER[0] + offsets,
    np.full(NUM_MIC, MIC_CENTER[1]),
    np.full(NUM_MIC, MIC_CENTER[2])
])

# ========== Angle settings ==========
ALL_AZ = np.arange(0, 181, 1)
EL_FIXED = 0


def load_audio(path, target_len=16000):
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
    return np.array([np.cos(el_rad) * np.cos(math_az_rad),
                     np.cos(el_rad) * np.sin(math_az_rad),
                     np.sin(el_rad)])


def generate_one_sample(num_src):
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
                src_path = random.choice(SRC_LIST)
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


def generate_split(split_name, split_samples):
    """Generate one split with balanced source counts 0/1/2/3."""
    if split_samples == 0:
        print(f"\nSkipping split '{split_name}' (0 samples).")
        return

    split_wav, split_lbl = SPLIT_DIRS[split_name]
    samples_per_class = split_samples // 4
    print(f"\n=== Generating split '{split_name}' ({split_samples} samples, {samples_per_class} per class) ===")

    local_idx = 0
    for num_src in range(4):
        print(f"  -> class num_src={num_src}: generating {samples_per_class} samples...")
        for j in range(1, samples_per_class + 1):
            local_idx += 1
            uid = f"{BASE}-{split_name}-{local_idx:05d}"

            wav_data, az_list = generate_one_sample(num_src)
            wavfile.write(split_wav / f"{uid}.wav", FS, wav_data)

            with open(split_lbl / f"{uid}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([num_src])
                for az in az_list:
                    az_out = Decimal(str(az)).quantize(Decimal("0.0"), ROUND_HALF_UP)
                    writer.writerow([az_out])

            if j % 250 == 0 or j == samples_per_class:
                print(f"     {j}/{samples_per_class} done for num_src={num_src}")


# ========== Run generation ==========
print("\nStarting data generation...")
t0 = time.perf_counter()

for split_name, split_samples in SPLITS.items():
    generate_split(split_name, split_samples)

t1 = time.perf_counter()
print(f"\nDone. Dataset root: {DATASET_ROOT}")
print(f"Total generated samples: {TOTAL_SAMPLES}")
print(f"Elapsed time: {t1 - t0:.1f} seconds")
