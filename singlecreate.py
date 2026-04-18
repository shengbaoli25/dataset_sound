# generate_single_source_data.py (v2 - 支持从多级子目录随机选取声源)

import time
from pathlib import Path
import numpy as np, csv, random
import pyroomacoustics as pra
from scipy.io import wavfile
from decimal import Decimal, ROUND_HALF_UP
import warnings
from scipy.io.wavfile import WavFileWarning

warnings.filterwarnings("ignore", category=WavFileWarning)
# ========== 1. 修改声源目录的定义 ==========
ROOT_OUT = Path.cwd()
# 将 SRC_ROOT 指向包含所有声源的根目录
SRC_ROOT = ROOT_OUT / "original_dataset"

# 使用 rglob 递归搜索所有子目录下的 .wav 文件
print(f"正在从 '{SRC_ROOT}' 及其子目录中搜索声源...")
SRC_LIST = list(SRC_ROOT.rglob("*.wav"))
if not SRC_LIST:
    # 如果没找到，也搜索 .WAV (大写后缀)
    SRC_LIST.extend(list(SRC_ROOT.rglob("*.WAV")))

assert len(SRC_LIST) >= 1, f"在 '{SRC_ROOT}' 及其子目录中没有找到任何 .wav 文件。"
print(f"找到 {len(SRC_LIST)} 个可用的声源文件。")

# ========== 输出目录 (保持不变) ==========
DATASET_NAME =  "../datacreate/dataset/single"
DST_WAV = ROOT_OUT / DATASET_NAME / "wav";
DST_WAV.mkdir(parents=True, exist_ok=True)
DST_LBL = ROOT_OUT / DATASET_NAME / "label";
DST_LBL.mkdir(parents=True, exist_ok=True)
BASE = "single"

# ========== 场景 / 麦克风阵列参数 (保持不变) ==========
ROOM_DIM = np.array([8., 8., 5.])
MIC_CENTER = np.array([4., 4., 2.5])
NUM_MIC = 4
DIAMETER = 0.08
SPACING = DIAMETER / (NUM_MIC - 1)
FS = 16000
SIG_DUR = 1.0
DIST_RANGE = (0.5, 2.5)
RT60_RANGE = (0.2, 1.2)

# ========== 构建线性阵列 (保持不变) ==========
offsets = np.linspace(SPACING * (NUM_MIC - 1) / 2, -SPACING * (NUM_MIC - 1) / 2, NUM_MIC)
MIC_COORDS = np.stack([
    MIC_CENTER[0] + offsets,
    np.full(NUM_MIC, MIC_CENTER[1]),
    np.full(NUM_MIC, MIC_CENTER[2])
])

# ========== 设置角度范围 (保持不变) ==========
ALL_AZ = np.arange(0, 181, 1)
el_fixed = 0


# ========== 音频加载函数 (保持不变) ==========
def load_audio(path, target_len=16000):
    try:
        sr, sig = wavfile.read(path)
    except Exception as e:
        print(f"\n警告: 读取文件失败 {path.name}: {e}. 跳过此文件。")
        return None

    if sr != FS:
        try:
            import resampy
            # 如果音频是多通道的，只取第一个通道
            if sig.ndim > 1:
                sig = sig[:, 0]
            sig = resampy.resample(sig, sr, FS)
        except ImportError:
            raise ImportError("请安装 `resampy` 库 (pip install resampy) 来处理采样率不匹配的问题。")

    if sig.dtype != np.float32:
        # 归一化前处理多通道情况
        if sig.ndim > 1:
            sig = sig[:, 0]
        # 检查是否为静音文件
        if np.max(np.abs(sig)) == 0:
            print(f"\n警告: 文件 {path.name} 为静音文件，可能导致错误。跳过。")
            return None
        max_val = np.iinfo(sig.dtype).max or float(np.max(np.abs(sig)))
        sig = sig.astype(np.float32) / max_val

    if len(sig) < target_len:
        sig = np.pad(sig, (0, target_len - len(sig)), 'wrap')
    else:
        sig = sig[:target_len]

    return sig


# ========== 数据生成 ==========
NUM_SAMPLES = 5000
print(f"准备生成 {NUM_SAMPLES} 个单声源样本...")
t0 = time.perf_counter()

for idx in range(1, NUM_SAMPLES + 1):
    num_src = 1
    az_deg = random.choice(ALL_AZ)

    rt60 = np.random.uniform(*RT60_RANGE)
    e_abs, max_order = pra.inverse_sabine(rt60, ROOM_DIM.tolist())
    room = pra.ShoeBox(ROOM_DIM, fs=FS,
                       materials=pra.Material(e_abs),
                       max_order=max_order,
                       air_absorption=True)
    room.add_microphone_array(pra.MicrophoneArray(MIC_COORDS, FS))

    # --- 从声源列表中随机选择并加载 ---
    sig = None
    while sig is None:  # 持续尝试直到成功加载一个有效的音频
        src_path = random.choice(SRC_LIST)
        sig = load_audio(src_path, target_len=int(SIG_DUR * FS))

    dist = np.random.uniform(*DIST_RANGE)

    az, el = np.deg2rad([az_deg, el_fixed])
    direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    src_pos = MIC_CENTER + dist * direction
    src_pos = np.clip(src_pos, 0.02, ROOM_DIM - 0.02)
    room.add_source(src_pos, signal=sig)

    # 模拟并保存
    room.simulate(snr=np.random.uniform(0, 30))
    uid = f"{BASE}-{idx:04d}"

    wav_data = (room.mic_array.signals.T * 32767).astype(np.int16)
    wavfile.write(DST_WAV / f"{uid}.wav", FS, wav_data)

    with open(DST_LBL / f"{uid}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        az_out = Decimal(str(az_deg)).quantize(Decimal('0.0'), ROUND_HALF_UP)
        dist_out = Decimal(str(dist)).quantize(Decimal('0.00'), ROUND_HALF_UP)
        writer.writerow([az_out, dist_out])

t1 = time.perf_counter()
print(f"\n✅ 单声源数据生成完成 -> {DST_WAV.parent}")
print(f"⏱️ 总耗时: {t1 - t0:.1f} 秒")