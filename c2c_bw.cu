// ============================================================
// gb300_test — DGX GB300 C2C / EGM 頻寬測試
// 開發者 (Developer): tyyang
// ============================================================
//
// c2c_bw.cu — Host <-> Device (NVLink-C2C) 頻寬測試
// 適用: Grace Hopper (GH200) / Grace Blackwell (GB10/GB300, DGX Spark/Station)
// 目的: 量測 H2D (Host->Device) / D2H (Device->Host) / 雙向 的 C2C 頻寬 (GB/s)
//
// 編譯:  nvcc -O3 -arch=native -o c2c_bw c2c_bw.cu
// 執行:  ./c2c_bw [--sizes "256MiB 512MiB 1GiB 2GiB 4GiB"] [--iters 20]
//
// 注意: 使用 pinned (page-locked) host memory + cudaMemcpyAsync,
//       這是跑在 NVLink-C2C 上、最接近真實 offload 讀取權重路徑的測法。

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GB (1024.0 * 1024.0 * 1024.0)

static void die(cudaError_t e, const char *msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error @ %s: %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

// 解析 "512MiB" / "1GiB" / "2GB" -> bytes
static size_t parse_size(const char *s) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s) return 0;
    if (strstr(end, "GiB") || strstr(end, "GB")) return (size_t)(v * 1024.0 * 1024.0 * 1024.0);
    if (strstr(end, "MiB") || strstr(end, "MB")) return (size_t)(v * 1024.0 * 1024.0);
    if (strstr(end, "KiB") || strstr(end, "KB")) return (size_t)(v * 1024.0);
    return (size_t)v;
}

static double bw_gbs(double bytes_total, double seconds) {
    return (bytes_total / GB) / seconds;
}

// 單向複製: 回傳 GB/s
static double bench_copy(cudaMemcpyKind kind, size_t bytes, int iters,
                         void *h, void *d, cudaStream_t s,
                         cudaEvent_t e0, cudaEvent_t e1) {
    void *dst = (kind == cudaMemcpyHostToDevice) ? d : h;
    void *src = (kind == cudaMemcpyHostToDevice) ? h : d;
    die(cudaMemcpyAsync(dst, src, bytes, kind, s), "warmup");
    die(cudaStreamSynchronize(s), "warmup sync");

    die(cudaEventRecord(e0, s), "rec e0");
    for (int i = 0; i < iters; i++)
        die(cudaMemcpyAsync(dst, src, bytes, kind, s), "copy");
    die(cudaEventRecord(e1, s), "rec e1");
    die(cudaEventSynchronize(e1), "sync e1");

    float ms = 0.0f;
    die(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
    return bw_gbs((double)bytes * iters, ms / 1000.0);
}

int main(int argc, char **argv) {
    int iters = 20;
    const char *sizes_arg = "64MiB 256MiB 512MiB 1GiB 2GiB 4GiB";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sizes") && i + 1 < argc) sizes_arg = argv[++i];
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("Usage: %s [--sizes \"256MiB 1GiB ...\"] [--iters N]\n", argv[0]);
            return 0;
        }
    }

    int dev = 0;
    die(cudaSetDevice(dev), "set device");
    cudaDeviceProp prop;
    die(cudaGetDeviceProperties(&prop, dev), "props");
    printf("GPU %d: %s | HBM %zu MiB | SM %d | CC %d.%d\n",
           dev, prop.name, prop.totalGlobalMem / (1024 * 1024),
           prop.multiProcessorCount, prop.major, prop.minor);

    // 解析 buffer 大小清單
    size_t sizes[64]; int n = 0;
    char *tmp = strdup(sizes_arg);
    for (char *tok = strtok(tmp, " "); tok && n < 64; tok = strtok(NULL, " "))
        if ((sizes[n] = parse_size(tok)) > 0) n++;
    free(tmp);
    if (n == 0) { fprintf(stderr, "no valid sizes\n"); return 1; }

    // 一次配置最大 buffer
    size_t maxsz = sizes[0];
    for (int i = 1; i < n; i++) if (sizes[i] > maxsz) maxsz = sizes[i];

    void *h = NULL, *d = NULL;
    die(cudaHostAlloc(&h, maxsz, cudaHostAllocDefault), "pinned host alloc");
    die(cudaMalloc(&d, maxsz), "device alloc");
    memset(h, 0xAB, maxsz);

    cudaStream_t s, s2;
    die(cudaStreamCreate(&s),  "stream");
    die(cudaStreamCreate(&s2), "stream2");
    cudaEvent_t e0, e1, e2;
    die(cudaEventCreate(&e0), "event0");
    die(cudaEventCreate(&e1), "event1");
    die(cudaEventCreate(&e2), "event2");

    printf("\n%-10s | %-14s | %-14s | %-14s | %-14s\n",
           "Size", "H2D GB/s", "D2H GB/s", "H2D+D2H GB/s", "雙向合計 GB/s");
    printf("-----------+----------------+----------------+----------------+----------------\n");

    for (int i = 0; i < n; i++) {
        size_t sz = sizes[i];

        double h2d = bench_copy(cudaMemcpyHostToDevice, sz, iters, h, d, s,  e0, e1);
        double d2h = bench_copy(cudaMemcpyDeviceToHost, sz, iters, h, d, s,  e0, e1);

        // 雙向: H2D 走 stream s, D2H 走 stream s2 併發
        die(cudaStreamSynchronize(s),  "pre-bidir sync s");
        die(cudaStreamSynchronize(s2), "pre-bidir sync s2");
        die(cudaEventRecord(e0, s), "bidir e0");
        for (int j = 0; j < iters; j++) {
            die(cudaMemcpyAsync(d, h, sz, cudaMemcpyHostToDevice, s),  "bidir h2d");
            die(cudaMemcpyAsync(h, d, sz, cudaMemcpyDeviceToHost, s2), "bidir d2h");
        }
        die(cudaEventRecord(e1, s),  "bidir h2d end");
        die(cudaEventRecord(e2, s2), "bidir d2h end");
        die(cudaEventSynchronize(e1), "bidir sync h2d");
        die(cudaEventSynchronize(e2), "bidir sync d2h");

        float ms1 = 0.0f, ms2 = 0.0f;
        die(cudaEventElapsedTime(&ms1, e0, e1), "bidir elapsed h2d");
        die(cudaEventElapsedTime(&ms2, e0, e2), "bidir elapsed d2h");
        double t = (ms1 > ms2 ? ms1 : ms2) / 1000.0;
        double bidir_total = bw_gbs((double)sz * iters * 2.0, t);

        char szstr[16];
        if (sz >= 1024ULL * 1024 * 1024)      snprintf(szstr, sizeof szstr, "%zuGiB", sz / (1024 * 1024 * 1024));
        else if (sz >= 1024ULL * 1024)        snprintf(szstr, sizeof szstr, "%zuMiB", sz / (1024 * 1024));
        else if (sz >= 1024ULL)               snprintf(szstr, sizeof szstr, "%zuKiB", sz / 1024);
        else                                  snprintf(szstr, sizeof szstr, "%zuB", sz);

        printf("%-10s | %-14.1f | %-14.1f | %-14.1f | %-14.1f\n",
               szstr, h2d, d2h, h2d + d2h, bidir_total);
    }

    die(cudaEventDestroy(e0), "destroy e0");
    die(cudaEventDestroy(e1), "destroy e1");
    die(cudaEventDestroy(e2), "destroy e2");
    die(cudaStreamDestroy(s),  "destroy s");
    die(cudaStreamDestroy(s2), "destroy s2");
    die(cudaFree(d), "free d");
    die(cudaFreeHost(h), "free h");
    return 0;
}
