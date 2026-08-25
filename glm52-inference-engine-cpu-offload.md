# GLM-5.2 推論引擎 CPU Offload 技術文件

**文件版本**：v1.0  
**參考環境**：ASUS Pro ET900N-G3 GB300 Station × 2 節點  
**模型**：nvidia/GLM-5.2-NVFP4（[HuggingFace](https://huggingface.co/nvidia/GLM-5.2-NVFP4)）  
**推論引擎**：vLLM v0.27.1 / SGLang + KT-Kernel  

---

## 目錄

1. [GLM-5.2 架構精確解析](#1-glm-52-架構精確解析)
2. [Decode Step 完整流程](#2-decode-step-完整流程)
3. [CPU Offload 機制說明](#3-cpu-offload-機制說明)
4. [EGM（Extended GPU Memory）分析](#4-egmextended-gpu-memory分析)
5. [vLLM 的 CPU Offload 實作](#5-vllm-的-cpu-offload-實作)
6. [SGLang + KT-Kernel 的改良](#6-sglang--kt-kernel-的改良)
7. [平台適用建議](#7-平台適用建議)
8. [附錄：術語解釋](#8-附錄術語解釋)

---

## 1. GLM-5.2 架構精確解析

### 1.1 基本規格（來自 config.json）

| 參數 | 值 | 說明 |
|------|-----|------|
| `num_hidden_layers` | **78** | 主要層數（Layer 0–77） |
| `num_nextn_predict_layers` | 1 | 額外 MTP 預測層（Layer 78） |
| `hidden_size` | 6,144 | 模型寬度 |
| `first_k_dense_replace` | **3** | 前 3 層為 Dense MLP |
| MoE 層數 | **75 層**（Layer 3–77） | `mlp_layer_types = sparse` |
| `n_routed_experts` | **256** 個/層 | 路由 expert 數量 |
| `n_shared_experts` | **1** 個/層 | 每層必激活的共享 expert |
| `num_experts_per_tok` | **8**（top-8） | 每個 token 激活的 routed expert 數 |
| 激活比例 | **3.1%**（8/256） | 每次推論只用極少數 expert |
| `moe_intermediate_size` | 2,048 | 每個 expert 的 FFN 寬度 |
| `intermediate_size` | 12,288 | Dense MLP（Layer 0–2）的 FFN 寬度 |
| 總參數 | ~743B | |
| 每次激活參數 | ~39B | |
| `model_type` | `glm_moe_dsa` | MLA + DSA 架構 |

### 1.2 Attention 規格（MLA - Multi-head Latent Attention）

| 參數 | 值 | 說明 |
|------|-----|------|
| `num_attention_heads` | 64 | |
| `q_lora_rank` | 2,048 | Q 壓縮維度 |
| `kv_lora_rank` | **512** | KV 壓縮維度（MLA 核心） |
| `qk_nope_head_dim` | 192 | 非 RoPE QK 維度 |
| `qk_rope_head_dim` | 64 | RoPE QK 維度 |
| `v_head_dim` | 256 | V 維度 |

### 1.3 層結構全覽

```
Layer 0–2   (3 層)：Dense MLP
            ├── Attention：MLA（BF16）
            ├── FFN：標準 Dense（intermediate=12288，BF16）
            └── DSA Indexer（BF16）

Layer 3–77  (75 層)：MoE (Sparse)
            ├── Attention：MLA（BF16）
            ├── Shared Expert × 1（BF16，每次必計算）
            ├── Router：sigmoid scoring，top-8 selection
            └── Routed Experts × 256（NVFP4，每次激活 8 個）

Layer 78    (1 層)：MTP（Multi-Token Prediction）
            └── 整層 BF16，跳過量化
```

### 1.4 NVFP4 量化規則

`quantization_config.ignore` 明確列出**保留 BF16** 的部分：

| 組件 | 精度 | 原因 |
|------|------|------|
| `embed_tokens`、`lm_head` | BF16 | Embedding 精度敏感 |
| Layer 0–2（整層） | BF16 | Dense 層，跳過量化 |
| Layer 3–77 的 `self_attn` | BF16 | Attention 精度敏感 |
| Layer 3–77 的 `mlp.shared_experts` | BF16 | 每層必激活，保留精度 |
| Layer 78（整層） | BF16 | MTP 層 |
| **Layer 3–77 的 `mlp.experts`（routed）** | **NVFP4** | 主體量化目標 |
| KV Cache | **FP8** | `kv_cache_scheme` 設定 |

**NVFP4 規格**：
- `num_bits: 4`，`type: float`，`group_size: 16`
- 每 16 個 4-bit weight 共用 1 個 scale factor
- 有效位元率：~0.5625 bytes/param（含 scale overhead）

### 1.5 記憶體分佈估算

```
GLM-5.2 NVFP4 記憶體組成：

BF16 部分（必須常駐 HBM）：
  MLA Attention × 78 層      ≈  26 GB
  Shared Expert × 75 層      ≈   6 GB
  Dense MLP × 3 層           ≈   1 GB
  Embedding + lm_head        ≈   4 GB
  DSA Indexer + MTP + Norm   ≈   6 GB
  ─────────────────────────────────
  BF16 小計                  ≈  43 GB

NVFP4 部分（可 offload 到 LPDDR5X）：
  Routed Experts（256×75層）
  256 × 75 × 3 × 6144 × 2048 × 0.5625 byte
                             ≈ 200 GB
  ─────────────────────────────────
  模型 weights 合計          ≈ 243 GB（與 HBM 實測 244 GB 吻合）

KV Cache（FP8，MLA 壓縮）：
  每 token/層 = kv_lora_rank(512) + qk_rope(64) = 576 bytes
  ─────────────────────────────────
  32K context  ≈  1.47 GB
  64K context  ≈  2.94 GB
  128K context ≈  5.89 GB   ← 極小！MLA 的優勢
  1M  context  ≈ 47.1  GB
```

---

## 2. Decode Step 完整流程

### 2.1 整體流程

```
輸入：position N 的 hidden state（batch B 個 token）
輸出：position N+1 的 logits → 取樣得到下一個 token

For layer = 0 to 77：
  Step A: RMSNorm
  Step B: DSA Indexer（選 token 子集，GLM-5.2 特有）
  Step C: MLA Attention（壓縮 KV，BF16）
  Step D: Residual Add
  Step E: RMSNorm
  Step F: MLP（Dense 或 MoE）
  Step G: Residual Add

最後：RMSNorm → lm_head → logits
```

### 2.2 Layer 0–2：Dense MLP（BF16，常駐 HBM）

```python
# 全部在 HBM 執行，無 offload
gate = silu(x @ W_gate)          # [B, 6144] @ [6144, 12288]
up   = x @ W_up                  # [B, 6144] @ [6144, 12288]
out  = (gate * up) @ W_down      # [B, 12288] @ [12288, 6144]
```

### 2.3 Layer 3–77：MoE 層詳細步驟

#### Step F-1：Router（在 HBM 執行，很輕量）

```python
# W_router 是小矩陣，常駐 HBM
router_logits = x @ W_router       # [B, 6144] @ [6144, 256] → [B, 256]

# scoring_func = "sigmoid"（GLM-5.2 特有，非 softmax）
scores = sigmoid(router_logits)    # [B, 256]

# topk_method = "noaux_tc"
top8_scores, top8_indices = topk(scores, k=8)  # 選最高的 8 個

# norm_topk_prob = true
top8_weights = normalize(top8_scores)           # 加權係數

# 結果：每個 token 得到 8 個 expert index + 對應權重
# decode 時 B=1（單 token），計算量極小
```

#### Step F-2：Shared Expert（BF16，常駐 HBM，每次必算）

```python
# 不受 Router 控制，永遠執行
shared_out = shared_expert_ffn(x)  # BF16，直接從 HBM 讀取
```

#### Step F-3：Routed Expert 計算（**CPU Offload 發生點**）

```python
# 根據 top8_indices 決定計算路徑：
for expert_idx in top8_indices:
    if expert_idx in HBM_cache:
        # 熱 expert：直接從 HBM 讀取計算
        out_i = hbm_expert[expert_idx](x)
    else:
        # 冷 expert：需從 LPDDR5X 載入（H2D）
        # 或在 CPU 上直接計算（KT-Kernel）
        ...
```

#### Step F-4：加權合併

```python
# routed_scaling_factor = 2.5（GLM-5.2 特有）
total_out = shared_out + sum(out_i * weight_i for i in top8) * 2.5
```

---

## 3. CPU Offload 機制說明

### 3.1 術語

| 縮寫 | 全名 | 方向 | 說明 |
|------|------|------|------|
| **H2D** | Host to Device | CPU RAM → GPU HBM | 把資料從主記憶體搬到顯卡 |
| **D2H** | Device to Host | GPU HBM → CPU RAM | 把資料從顯卡搬回主記憶體 |
| **D2D** | Device to Device | GPU ↔ GPU | GPU 間傳輸 |

### 3.2 為什麼需要 CPU Offload

```
問題：GLM-5.2 NVFP4 Routed Experts 總大小 ≈ 200 GB
      單張 GB300 HBM 只有 256 GB
      
現有 TP=2 部署（無 offload）：
  每張 GPU 分擔一半 experts：128 × 75 × 18.87 MB ≈ 181 GB
  加 BF16 部分 ≈ 43 GB
  模型 weights 合計 ≈ 224 GB < 256 GB ✓ 勉強放得下
  剩餘 KV Cache：256 × 0.95 - 224 ≈ 19 GB（偏少）

啟用 Expert Offload 後（每張 GPU 保留 64 個 hot experts）：
  64 experts × 75 層 × 18.87 MB ≈ 90 GB（熱 expert）
  加 BF16 部分 ≈ 43 GB
  模型 weights ≈ 133 GB
  剩餘 KV Cache：256 × 0.85 - 133 ≈ 84 GB（4× 提升）
```

### 3.3 Expert Cache 管理（LRU 策略）

```
HBM Expert Cache Pool（熱 expert）：
  ┌─────────────────────────────────────┐
  │ Expert 23：高頻激活  ← 常駐 HBM    │
  │ Expert 71：高頻激活  ← 常駐 HBM    │
  │ Expert 89：最近使用  ← 在 cache    │
  │ ...（共 N 個熱 expert）             │
  └─────────────────────────────────────┘
         ↑ evict（LRU）
         ↓ load（H2D）
LPDDR5X Expert Pool（冷 expert）：
  Expert 1, 2, 4, 5, ...（低頻 expert）
```

---

## 4. EGM（Extended GPU Memory）分析

### 4.1 什麼是 EGM

EGM 是 NVIDIA GB300 Grace Blackwell 架構的 BIOS 設定。

```
GB300 內部連接（EGM Enable）：

Grace CPU（Neoverse-V2, 72核）
    │
    │ C2C Link（~900 GB/s）← 關鍵頻寬
    │ 低延遲，類 NVLink
    │
Blackwell GPU（B200）
    │
    HBM3e（256 GB，~4 TB/s）

LPDDR5X（744 GiB，~273 GB/s）
  ↑ 由 Grace CPU 管理
  ↑ EGM Enable 時 GPU 可直接定址
```

**EGM Enable**：GPU 透過 C2C 直接存取 LPDDR5X，無需經過 PCIe  
**EGM Disable**：GPU 存取 LPDDR5X 需走 PCIe（慢 14 倍）

### 4.2 H2D 頻寬對比

| 路徑 | 頻寬 | 傳輸 18.87 MB（1 expert） | 傳輸 11.3 GB（1 decode step） |
|------|------|--------------------------|-----------------------------|
| C2C（EGM Enable） | ~900 GB/s | **0.02 ms** | **12.6 ms** |
| PCIe 5.0 x16（EGM Disable） | ~64 GB/s | **0.30 ms** | **176.9 ms** |
| 差異 | **14×** | | |

### 4.3 EGM 對不同場景的效益

| 場景 | EGM 效益 | 說明 |
|------|---------|------|
| **vLLM layer-level offload** | 🟡 中等 | H2D 整層加速（但整層傳輸仍是浪費） |
| **vLLM per-expert offload（未實作）** | ✅ 高 | 冷 expert H2D 從 177ms → 12.6ms |
| **SGLang + KT-Kernel** | ❌ 低 | 冷 expert 在 CPU 計算，根本不 H2D |
| **模型啟動載入** | 🟡 輕微 | runai_streamer 可能利用 C2C 路徑 |
| **純 HBM 推論（無 offload）** | ❌ 無 | 推論在 HBM 內，不涉及 C2C |

### 4.4 EGM 真正有效的使用條件

```
EGM 發揮最大效益：

BIOS: EGM Enable                                    ✓
vLLM: --cpu-offload-gb N（啟用 weight offload）     ✓
vLLM: 實作 per-expert 粒度（目前未實作）            ✗

→ 目前 EGM + vLLM 的實際效益有限
→ 如果未來 vLLM 實作 per-expert offload，EGM 將是關鍵加速器
```

---

## 5. vLLM 的 CPU Offload 實作

### 5.1 `--cpu-offload-gb` 的真實機制

> ⚠️ **重要釐清**：`--cpu-offload-gb` 卸載的是**模型權重（Model Weights）**，不是 KV Cache。

```
vLLM offload 機制（Layer-level，PR #6496 + PR #29941）：

平時（idle）：
  指定的 Transformer layers 的 weights 存放在
  CPU pinned memory（LPDDR5X）

Forward pass 執行時（layer N）：
  Step 1：layer N weights：CPU → GPU（H2D，prefetch）
  Step 2：layer N-1 weights：GPU 計算完後 → CPU（D2H evict）
  Step 3：同時 prefetch layer N+1 weights（pipeline overlap）

效果：
  HBM 占用 = 完整模型 - offload 的部分
  代價 = 每層 forward 前等待 H2D 傳輸
         （可被 prefetch pipeline 部分掩蓋）
```

### 5.2 Layer-level offload 對 MoE 的問題

```
MoE 每層有 256 experts，但每次只用 top-8（3.1%）

vLLM layer-level offload 的行為：
  當 layer N offload 到 CPU 時：
  → H2D：整層所有 256 experts × 18.87 MB = 4.78 GB
  → 只使用其中 8 個 experts = 151 MB
  → 浪費傳輸：4.63 GB（96.9% 是無效傳輸！）
```

### 5.3 推論延遲影響（GLM-5.2，layer-level offload）

```
每個 decode step（75 MoE 層）：

EGM Disable（PCIe 64 GB/s）：
  每層 H2D：4.78 GB / 64 GB/s = 74.7 ms
  75 層合計：75 × 74.7 ms = 5,600 ms（有 prefetch pipeline 可部分掩蓋）
  → 嚴重影響延遲 🔴

EGM Enable（C2C 900 GB/s）：
  每層 H2D：4.78 GB / 900 GB/s = 5.3 ms
  75 層合計：有效 pipeline 後開銷大幅降低
  → 可接受（但仍有 96.9% 傳輸浪費）🟡
```

### 5.4 目前 vLLM 建議參數（EGM Enable 環境）

```bash
vllm serve /models/glm5.2-nvfp4/latest \
  --max-model-len=131072 \
  --max-num-seqs=64 \
  --tensor-parallel-size=2 \
  --gpu-memory-utilization=0.85 \    # 降低（從 0.95），釋放空間給 offload
  --cpu-offload-gb=32 \              # 從 LPDDR5X offload 32 GB
  --kv-cache-dtype=fp8 \
  --speculative-config.method=dspark \
  --enable-prefix-caching \
  --load-format=runai_streamer \
  --model-loader-extra-config='{"memory_limit":0}'
```

> **注意**：Issue #14233 確認 SGLang 尚未實作 per-expert 粒度 offload（2025-12 提出）。vLLM 同樣是 layer-level。

---

## 6. SGLang + KT-Kernel 的改良

### 6.1 核心改良：Per-Expert 粒度 + CPU 計算

```
SGLang + KT-Kernel 的架構：

GPU HBM                           CPU LPDDR5X
┌───────────────────┐             ┌────────────────────┐
│ Hot Experts       │             │ Cold Experts       │
│（高頻 expert）     │             │（低頻 expert）      │
│ 直接在 GPU 計算   │             │ 直接在 CPU 計算     │
│（HBM 讀取）       │  並行執行   │（AMX/AVX-512 GEMM）│
└───────────────────┘             └────────────────────┘
        ↓                                  ↓
    加權合併兩側輸出（CPU → GPU 只傳結果，非 weights）
```

**關鍵差異**：不是把冷 expert weights H2D 到 GPU 再算，而是**冷 expert 直接在 CPU 上計算**，只把**計算結果**（小張量）傳回 GPU 合併。

### 6.2 Decode Step 對比

```
同一個 decode step，top-8 中有 3 個 hot（GPU），5 個 cold（CPU）：

vLLM layer-level offload：
  1. Router → top-8 indices
  2. H2D：整層 256 experts（4.78 GB！）
  3. GPU 計算 8 個，其他 248 個白載
  4. D2H：整層搬回
  ← 無效傳輸 96.9%

SGLang + KT-Kernel per-expert hybrid：
  1. Router → top-8 indices
  2. 查詢 GPU expert mask
  3a. GPU stream：
        計算 expert 23, 71, 89（從 HBM 直接讀）
  3b. CPU threads（並行）：
        AMX GEMM 計算 expert 1, 15, 104, 156, 200
  4. 合併：CPU 結果（很小）→ GPU，加權求和
  ← H2D 傳輸量：≈ 0（只傳計算結果）
```

### 6.3 四種 Expert 調度策略

| 策略 | 說明 | 適用場景 |
|------|------|---------|
| `uniform` | 每層均勻分配 N 個 GPU expert | 預設，無需統計資料 |
| `frequency` | 把最常被激活的 expert 放 GPU | **最佳效能**，需預先收集統計 |
| `front-loading` | 從第一層開始填滿 GPU | 測試基準 |
| `random` | 隨機選（seed=42） | 對比基準 |

### 6.4 動態 Expert 更新（`--kt-enable-dynamic-expert-update`）

```
Prefill 階段（長 prompt 輸入）：
  統計每個 expert 的激活頻率
  → 動態重排 GPU expert mask
  → 把高頻 expert 換到 GPU HBM

Decode 階段：
  使用更新後的 placement，GPU 命中率更高
  CPU 需計算的 cold experts 更少
  → 整體延遲降低
```

### 6.5 實測效能（Qwen3-Next-80B，4× RTX 4090）

| GPU Expert 比例 | uniform | frequency | **dynamic-update** | 純 GPU（基準） |
|----------------|---------|-----------|-------------------|--------------|
| 0%（全 CPU） | 53 t/s | 53 t/s | 53 t/s | — |
| 10% | 57 t/s | 59 t/s | **70 t/s** | — |
| 30% | 62 t/s | 67 t/s | **76 t/s** | — |
| 50% | 65 t/s | 76 t/s | **81 t/s** | — |
| 70% | 76 t/s | 89 t/s | **89 t/s** | — |
| 100% | 112 t/s | 114 t/s | 113 t/s | **112 t/s** |

> **30% GPU experts + dynamic update = 純 GPU 的 68% 速度**，但 HBM 用量只需 30%！

### 6.6 GLM-5.2 官方 KT-Kernel 啟動指令

```bash
# GLM-5.2 FP8，TP=8，8-GPU 伺服器
python -m sglang.launch_server \
  --model-path /path/to/GLM-5.2-FP8 \
  --kt-weight-path /path/to/GLM-5.2-FP8 \
  --kt-cpuinfer 96 \                       # CPU inference threads
  --kt-threadpool-count 2 \                # CPU thread pools
  --kt-num-gpu-experts 30 \                # 每層 30/256 experts 在 GPU
  --kt-method FP8 \
  --kt-gpu-prefill-token-threshold 1024 \  # prefill 觸發動態更新
  --kt-enable-dynamic-expert-update \
  --kt-expert-placement-strategy uniform \
  --tp-size 8 \
  --attention-backend nsa \
  --kv-cache-dtype fp8_e4m3 \
  --disable-shared-experts-fusion \
  --tool-call-parser glm47 \
  --reasoning-parser glm45
```

---

## 7. 平台適用建議

### 7.1 硬體規格（參考環境）

| 節點 | CPU | RAM | GPU | 網路 |
|------|-----|-----|-----|------|
| gb300-master | ARM Neoverse-V2 72核 | 744 GiB LPDDR5X | NVIDIA GB300（256 GB HBM3e） | ConnectX-8 400 GbE |
| gb300-worker-01 | ARM Neoverse-V2 72核 | 744 GiB LPDDR5X | NVIDIA GB300（256 GB HBM3e） | ConnectX-8 400 GbE |

### 7.2 現有部署（vLLM v0.27.1，TP=2，無 offload）

```bash
vllm serve /models/afsbox/nvidia/glm5.2-nvfp4/latest \
  --max-model-len=131072 \
  --max-num-seqs=64 \
  --tensor-parallel-size=2 \
  --pipeline-parallel-size=1 \
  --nnodes=2 \
  --gpu-memory-utilization=0.95 \
  --max-num-batched-tokens=32768 \
  --kv-cache-dtype=fp8 \
  --speculative-config.method=dspark \
  --speculative-config.num_speculative_tokens=7 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-force-include-usage \
  --load-format=runai_streamer
```

**現況**：
- 每張 GPU 負責 128 experts（TP=2 分割）
- HBM 使用：244/256 GB（95%）
- KV Cache：約 19 GB（128K context 最多 ~3 個並發請求）
- Prefix Cache 命中率：62%（良好）

### 7.3 各方案對比

| 方案 | HBM 模型佔用 | KV Cache 空間 | 效能 | 複雜度 | EGM 效益 |
|------|------------|--------------|------|--------|---------|
| **現有 TP=2，無 offload** | 224 GB/GPU | ~19 GB | 12.8 t/s | 低 | 無 |
| **vLLM + cpu-offload-gb（EGM ✗）** | 可降低 | 增加 | 大幅下降 | 低 | — |
| **vLLM + cpu-offload-gb（EGM ✓）** | 可降低 | 增加 | 輕微下降 | 低 | 🟡 中等 |
| **SGLang + KT-Kernel（ARM 待驗證）** | 可大幅降低 | 大幅增加 | 需測試 | 高 | ❌ 低 |

### 7.4 EGM + vLLM 建議設定

若 BIOS 啟用 EGM，搭配以下參數：

```bash
# 調整方向：
--gpu-memory-utilization=0.80        # 從 0.95 降低，釋放空間給 offload
--cpu-offload-gb=32                  # 允許 32 GB offload 到 LPDDR5X
--max-model-len=196608               # 可嘗試突破 131072（利用 EGM 擴展）
--max-num-seqs=128                   # 從 64 提升（KV Cache 更充裕）

# 預期效果：
# HBM weights：約 210 GB → 節省 33 GB
# KV Cache：約 50 GB（從 19 GB 提升 2.6×）
# 推論速度：輕微下降（prefetch pipeline 掩蓋大部分延遲）
```

### 7.5 SGLang + KT-Kernel ARM 注意事項

```
⚠️ KT-Kernel 主要針對 x86 CPU 優化：
   - Intel Sapphire Rapids：AMX（最快）
   - AMD EPYC：AVX-512
   
ARM Neoverse-V2（你的平台）：
   - 有 SVE2 / NEON 向量指令
   - KT-Kernel 是否支援 ARM 需要確認
   - 若 CPU GEMM 效率低，expert offload 效益有限

建議：
   先測試 kt version 是否支援 ARM
   再評估是否值得切換到 SGLang + KT
```

---

## 8. 附錄：術語解釋

| 術語 | 說明 |
|------|------|
| **H2D** | Host to Device，CPU RAM → GPU HBM 傳輸 |
| **D2H** | Device to Host，GPU HBM → CPU RAM 傳輸 |
| **MoE** | Mixture of Experts，稀疏激活的 FFN 架構 |
| **MLA** | Multi-head Latent Attention，壓縮 KV 的注意力機制 |
| **DSA** | DeepSeek Sparse Attention，稀疏 token 選取機制 |
| **MTP** | Multi-Token Prediction，一次預測多個 token |
| **NVFP4** | NVIDIA FP4 量化，4-bit float，group_size=16 |
| **EGM** | Extended GPU Memory，BIOS 設定讓 GPU 可直接定址 CPU RAM |
| **C2C** | Chip-to-Chip，GB300 中 Grace CPU 與 Blackwell GPU 的直連 |
| **KV Cache** | Key-Value Cache，Attention 的歷史 token 快取 |
| **TP** | Tensor Parallelism，將模型張量分割到多 GPU |
| **LRU** | Least Recently Used，最久未使用的淘汰策略 |
| **AMX** | Advanced Matrix Extensions，Intel CPU 矩陣運算加速指令 |
| **prefetch** | 預取，在需要前提前將資料搬移到目標位置 |
| **top-k routing** | MoE 中每個 token 選 k 個 expert 計算 |
| **hot expert** | 被 router 頻繁選中的 expert（常駐 HBM） |
| **cold expert** | 被 router 低頻選中的 expert（offload 到 LPDDR5X） |
| **DSpark** | NVIDIA/ASUS 專用的 Speculative Decoding 方案 |
| **Runai Streamer** | Run:ai 開發的模型快速串流載入工具 |

---

## 附錄 B：vLLM vs SGLang+KT 完整對比

| 面向 | vLLM v0.27.1 | SGLang + KT-Kernel |
|------|-------------|-------------------|
| **offload 粒度** | Layer-level（整層 256 experts） | **Per-expert（只處理 top-k）** |
| **冷 expert 執行位置** | H2D 搬到 GPU 再計算 | **直接在 CPU 上計算** |
| **H2D 傳輸量/step** | 整層 4.78 GB | ≈ 0（只傳計算結果） |
| **GPU/CPU 並行** | 無 | **有**（GPU 算熱，CPU 算冷） |
| **Expert placement 策略** | 無 | uniform / frequency / dynamic |
| **動態 placement 更新** | 無 | ✅ prefill 後自動調整 |
| **EGM 效益** | 🟡 中等（H2D 加速） | ❌ 低（根本不 H2D） |
| **ARM 支援** | ✅ | ⚠️ 主要 x86，ARM 待確認 |
| **GLM-5.2 NVFP4 支援** | ✅（目前部署） | ✅（官方文件有） |
| **部署複雜度** | 低 | 高（需安裝 KT-Kernel） |
| **per-expert offload 議題** | Issue 尚未實作 | ✅ 已實作（KT-Kernel） |

---

## 附錄 C：KV Cache 大小快速查詢（GLM-5.2，MLA，FP8）

公式：`KV_bytes = (kv_lora_rank + qk_rope_head_dim) × num_layers × 1 byte × tokens`  
= `(512 + 64) × 78 × 1 × tokens`  
= `576 × 78 × tokens`

| Context 長度 | KV Cache 大小 | 備註 |
|------------|-------------|------|
| 32K tokens | 1.47 GB | 輕量 |
| 64K tokens | 2.94 GB | 一般對話 |
| **128K tokens** | **5.89 GB** | **現有 max_model_len** |
| 256K tokens | 11.78 GB | EGM offload 後可考慮 |
| 512K tokens | 23.55 GB | |
| 1M tokens | 47.11 GB | 理論上限 |

> **MLA 的 KV Cache 比標準 Attention 小 ~57 倍**（標準版 128K 需 ~312 GB）  
> 這使得 KV Cache 幾乎不是瓶頸，真正的瓶頸是模型 weights 的 HBM 佔用。

---

*文件結束*  
*生成時間：2026-08-20*  
*參考：nvidia/GLM-5.2-NVFP4 config.json、vLLM PR #6496 #29941、KTransformers Expert Scheduling Tutorial*
