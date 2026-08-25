#!/usr/bin/env bash
# run_egm_bench.sh — 收集 EGM 狀態 + 量測 H2D/D2H C2C 頻寬
# 在 DGX (Grace Hopper / Grace Blackwell) 上執行。EGM on/off 是 BIOS 設定,
# 需在 BIOS 改好後重開機, 再各跑一次本腳本, 比對兩個結果檔。
#
# 用法:
#   ./run_egm_bench.sh            # 用已編譯的 c2c_bw (若存在) + nvbandwidth(若有)
#   ./run_egm_bench.sh --build    # 先 nvcc 編譯 c2c_bw.cu
#   ./run_egm_bench.sh --nvbw /path/to/nvbandwidth   # 指定 nvbandwidth 路徑

set -u

# ---------- 參數 ----------
BUILD=0
NVBW=""
while [ $# -gt 0 ]; do
  case "$1" in
    --build) BUILD=1 ;;
    --nvbw) NVBW="${2:-}"; shift ;;
    --iters) ITERS="${2:-20}"; shift ;;
    *) echo "未知參數: $1"; exit 2 ;;
  esac
  shift
done
ITERS="${ITERS:-20}"

DIR="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$DIR/result-$STAMP.txt"

# ---------- 1. 系統 / EGM 狀態 ----------
{
echo "==================== EGM / C2C 狀態快照 ===================="
echo "時間: $(date)"
echo "主機: $(hostname)"
echo "架構: $(uname -m) | 核心: $(uname -r)"

echo "---- GPU 列表 ----"
nvidia-smi -L 2>/dev/null || echo "(無 nvidia-smi)"

echo "---- GPU 型號 / 驅動 ----"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv 2>/dev/null

echo "---- C2C / NVLink 拓樸 ----"
nvidia-smi topo -m 2>/dev/null

echo "---- EGM carveout 偵測 (啟發式) ----"
echo "MemTotal (GB, 被 carveout 吃掉會比實體 DRAM 小):"
awk '/MemTotal/{printf "  %.1f GB\n", $2/1024/1024}' /proc/meminfo
echo "HugePages / 1GB hugepage 狀態:"
grep -E 'HugePages_(Total|Free)' /proc/meminfo 2>/dev/null || echo "  (無)"
echo "nvgrace-egm 字元裝置 (存在代表 EGM 驅動已掛載):"
ls -l /dev/egm* 2>/dev/null || echo "  (無 /dev/egmX — EGM 驅動未載入或核心過舊)"
echo "sysfs egm 屬性:"
find /sys -maxdepth 4 -iname '*egm*' 2>/dev/null | head -20 || echo "  (無)"
echo "dmesg 中的 egm/carveout 訊息:"
dmesg 2>/dev/null | grep -iE 'egm|carveout|nvgrace' | tail -20 || echo "  (無, 或無權限讀 dmesg)"

echo
echo "==================== 記憶體頻寬基準 ===================="
} | tee "$OUT"

# ---------- 2. 自訂 CUDA 腳本 ----------
if [ "$BUILD" = "1" ]; then
  echo ">>> 編譯 c2c_bw.cu ..." | tee -a "$OUT"
  nvcc -O3 -arch=native -o "$DIR/c2c_bw" "$DIR/c2c_bw.cu" || { echo "編譯失敗" | tee -a "$OUT"; exit 1; }
fi

if [ -x "$DIR/c2c_bw" ]; then
  echo ">>> 自訂 CUDA 頻寬測試 (pinned memcpy, C2C)" | tee -a "$OUT"
  "$DIR/c2c_bw" --iters "$ITERS" 2>&1 | tee -a "$OUT"
else
  echo ">>> 未找到 $DIR/c2c_bw, 請先跑 --build" | tee -a "$OUT"
fi

# ---------- 3. nvbandwidth (若有) ----------
if [ -n "$NVBW" ] || command -v nvbandwidth >/dev/null 2>&1; then
  NVBW_BIN="${NVBW:-$(command -v nvbandwidth)}"
  echo ">>> nvbandwidth host<->device 測試 (本機 C2C, 用 numactl 鎖 NUMA)" | tee -a "$OUT"
  # 每個 NUMA 節點各跑一次, 區分 local C2C 與 remote socket
  for node in $(ls -d /sys/devices/system/node/node* 2>/dev/null | sed 's/.*node//'); do
    echo "---- NUMA node $node ----" | tee -a "$OUT"
    numactl --cpunodebind="$node" --membind="$node" \
      "$NVBW_BIN" -t host_to_device_memcpy_ce device_to_host_memcpy_ce \
                  host_to_device_bidirectional_memcpy_ce \
                  host_to_device_memcpy_sm device_to_host_memcpy_sm \
      2>&1 | tee -a "$OUT"
  done
else
  echo ">>> 未找到 nvbandwidth (可用 --nvbw 指定路徑; 或 git clone https://github.com/NVIDIA/nvbandwidth)" | tee -a "$OUT"
fi

echo | tee -a "$OUT"
echo "完成。結果檔: $OUT" | tee -a "$OUT"
echo "==> EGM off 與 on 各跑一次, 用 diff 比對兩個 result-*.txt。" | tee -a "$OUT"
