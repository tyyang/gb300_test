# gb300_test — DGX GB300 C2C / EGM 頻寬測試

> 開發者 (Developer): **tyyang**

量測 Grace Blackwell（DGX Station GB300）上 Host↔Device 的 **NVLink-C2C**
頻寬（H2D / D2H / 雙向），並收集 **EGM (Extended GPU Memory)** 狀態快照，
用於對比 EGM on/off。

## 檔案

| 檔案 | 說明 |
|------|------|
| `c2c_bw.cu` | 自訂 CUDA 頻寬測試（pinned memcpy + `cudaMemcpyAsync` + CUDA events） |
| `run_egm_bench.sh` | wrapper：抓 EGM 狀態 + 跑 CUDA 測試 +（可選）nvbandwidth，存 timestamp 結果 |

## 用法（在 DGX 上執行）

```bash
# 可選：抓官方 nvbandwidth
git clone https://github.com/NVIDIA/nvbandwidth && cd nvbandwidth && cmake . && make -j

# 編譯 + 跑
./run_egm_bench.sh --build --nvbw /path/to/nvbandwidth

# EGM on/off 對比
#   1. BIOS 關 EGM（carveout=0）→ 重開機 → 跑一次
#   2. BIOS 開 EGM（設 carveout）→ 重開機 → 再跑一次
#   3. diff result-*.txt
```

## 重點提醒

- **EGM 是 BIOS/UEFI 選項**，不是 runtime 開關，改完需重開機。
- EGM carveout 對 Host OS 隱形（`/proc/meminfo` MemTotal 會變小）。
- 一般 pinned memory 走的是 C2C coherent 路徑（過 IOMMU），與 EGM 開關無關；
  要真正測到 EGM 無 IOMMU 路徑，需從 `/dev/egmX` 或 VM 內配置。
- 參考值（GH200）：H2D ~380 GB/s、D2H ~297 GB/s、雙向 C2C 峰值 450 GB/s/方向。

## 參考

- [nvgrace-egm driver (LWN)](https://lwn.net/Articles/1081287/)
- [2x GH200 memory paths benchmark](https://dnhkng.github.io/posts/gh200-benchmarking/)
- [nvbandwidth](https://github.com/NVIDIA/nvbandwidth)
