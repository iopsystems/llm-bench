import type { GpuSpec } from '../engine/types'

// Peak numbers from vendor datasheets. Dense compute (no sparsity).
// FP16/BF16 columns reflect Tensor Core peak.
export const GPUS: GpuSpec[] = [
  {
    id: 'h100', name: 'NVIDIA H100', vendor: 'NVIDIA', family: 'Hopper',
    variants: [
      {
        id: 'sxm-80', label: 'SXM 80GB', hbmCapacityGB: 80,
        operatingPoints: [
          {
            id: 'peak', label: 'Peak',
            tflops: { fp16: 989, bf16: 989, fp8: 1979, int8: 1979 },
            hbmBandwidthGBs: 3350
          },
          {
            id: 'achievable', label: 'Achievable',
            // mamf-finder MAMF table, BF16: 794.5 TFLOPS, FP8: 1402.6 TFLOPS.
            // PyTorch torch.mm brute-force shape search; BF16 @ 2048×2048×13312,
            // FP8 @ 1024×9216×14336; torch 2.7.0+cu126 / 2.7.1+cu128.
            tflops: { fp16: 795, bf16: 795, fp8: 1403 },
            hbmBandwidthGBs: 3350,
            tflopsSources: ['mamf-finder'],
            notes: 'mamf-finder PyTorch torch.mm sweep; HBM not separately measured (using peak)'
          }
        ]
      },
      {
        id: 'pcie-80', label: 'PCIe 80GB', hbmCapacityGB: 80,
        operatingPoints: [
          {
            id: 'peak', label: 'Peak',
            tflops: { fp16: 756, bf16: 756, fp8: 1513, int8: 1513 },
            hbmBandwidthGBs: 2000
          },
          {
            id: 'achievable', label: 'Achievable',
            // Microbenchmark measured on H800 PCIe (H100 PCIe export variant).
            // wgmma dense FP16 ~703-729 TFLOPS, FP8/INT8 ~1440 TFLOPS.
            // Global memory test ~1861 GB/s on H800's 2039 GB/s HBM2e;
            // proportionally ~91% of H100 PCIe's 2000 GB/s peak.
            tflops: { fp16: 729, bf16: 729, fp8: 1448, int8: 1448 },
            hbmBandwidthGBs: 1820,
            tflopsSources: ['arxiv-2501-12084'],
            bandwidthSources: ['arxiv-2501-12084'],
            asOf: '2025-01',
            notes: 'Measured on H800 PCIe (same silicon as H100 PCIe, export-restricted NVLink); wgmma + global memory tests'
          }
        ]
      },
      {
        id: 'pcie-94', label: 'PCIe 94GB', hbmCapacityGB: 94,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 756, bf16: 756, fp8: 1513, int8: 1513 },
          hbmBandwidthGBs: 2400
        }]
      },
      {
        id: 'nvl-188', label: 'NVL (per GPU 94GB)', hbmCapacityGB: 94,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 989, bf16: 989, fp8: 1979, int8: 1979 },
          hbmBandwidthGBs: 3900
        }]
      }
    ]
  },
  {
    id: 'h200', name: 'NVIDIA H200', vendor: 'NVIDIA', family: 'Hopper',
    variants: [{
      id: 'sxm-141', label: 'SXM 141GB', hbmCapacityGB: 141,
      operatingPoints: [
        {
          id: 'peak', label: 'Peak',
          tflops: { fp16: 989, bf16: 989, fp8: 1979, int8: 1979 },
          hbmBandwidthGBs: 4800
        },
        {
          id: 'achievable', label: 'Achievable',
          // mamf-finder MAMF table. FP8 directly measured @ 1280×4096×12032,
          // torch 2.7.1+cu128: 1453.4 TFLOPS. BF16 inferred from author note
          // "H200 is the same" on the H100 SXM row (794.5 TFLOPS) — not a
          // separately measured H200 BF16 run.
          tflops: { fp16: 795, bf16: 795, fp8: 1453 },
          hbmBandwidthGBs: 4800,
          tflopsSources: ['mamf-finder'],
          notes: 'FP8 directly measured; BF16 inferred from mamf-finder author note "H200 is the same" as H100 SXM, not a separate H200 BF16 measurement'
        }
      ]
    }]
  },
  {
    id: 'a100', name: 'NVIDIA A100', vendor: 'NVIDIA', family: 'Ampere',
    variants: [
      {
        id: 'sxm-40', label: 'SXM 40GB', hbmCapacityGB: 40,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 312, bf16: 312, int8: 624 },
          hbmBandwidthGBs: 1555
        }]
      },
      {
        id: 'sxm-80', label: 'SXM 80GB', hbmCapacityGB: 80,
        operatingPoints: [
          {
            id: 'peak', label: 'Peak',
            tflops: { fp16: 312, bf16: 312, int8: 624 },
            hbmBandwidthGBs: 2039
          },
          {
            id: 'achievable', label: 'Achievable',
            // mamf-finder MAMF BF16: 271.2 TFLOPS @ 1024×10240×5120,
            // torch 2.6.0+cu126. 80GB variant per memory spec table in same
            // document.
            tflops: { fp16: 271, bf16: 271 },
            hbmBandwidthGBs: 2039,
            tflopsSources: ['mamf-finder'],
            notes: 'mamf-finder PyTorch torch.mm sweep, 80GB variant (SXM4/SXM5 not distinguished in the source); HBM not separately measured'
          }
        ]
      },
      {
        id: 'pcie-40', label: 'PCIe 40GB', hbmCapacityGB: 40,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 312, bf16: 312, int8: 624 },
          hbmBandwidthGBs: 1555
        }]
      },
      {
        id: 'pcie-80', label: 'PCIe 80GB', hbmCapacityGB: 80,
        operatingPoints: [
          {
            id: 'peak', label: 'Peak',
            tflops: { fp16: 312, bf16: 312, int8: 624 },
            hbmBandwidthGBs: 1935
          },
          {
            id: 'achievable', label: 'Achievable',
            // mamf-finder MAMF BF16: 252.9 TFLOPS @ 2048×5120×6144,
            // torch 2.5.1+cu124. 80GB variant per memory spec table.
            tflops: { fp16: 253, bf16: 253 },
            hbmBandwidthGBs: 1935,
            tflopsSources: ['mamf-finder'],
            notes: 'mamf-finder PyTorch torch.mm sweep, 80GB variant; HBM not separately measured'
          }
        ]
      }
    ]
  },
  {
    id: 'l40s', name: 'NVIDIA L40S', vendor: 'NVIDIA', family: 'Ada Lovelace',
    variants: [{
      id: 'pcie-48', label: 'PCIe 48GB', hbmCapacityGB: 48,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 362, bf16: 362, fp8: 733, int8: 733 },
        hbmBandwidthGBs: 864
      }]
    }]
  },
  {
    id: 'rtx-5090', name: 'NVIDIA RTX 5090', vendor: 'NVIDIA', family: 'Blackwell',
    variants: [{
      id: 'sku', label: '32GB', hbmCapacityGB: 32,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 209, bf16: 209, fp8: 419, int8: 419 },
        hbmBandwidthGBs: 1792
      }]
    }]
  },
  {
    id: 'rtx-5080', name: 'NVIDIA RTX 5080', vendor: 'NVIDIA', family: 'Blackwell',
    variants: [{
      id: 'sku', label: '16GB', hbmCapacityGB: 16,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 112.6, bf16: 112.6, fp8: 225, int8: 225 },
        hbmBandwidthGBs: 960
      }]
    }]
  },
  {
    id: 'rtx-4090', name: 'NVIDIA RTX 4090', vendor: 'NVIDIA', family: 'Ada Lovelace',
    variants: [{
      id: 'sku', label: '24GB', hbmCapacityGB: 24,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 165, bf16: 165, fp8: 330, int8: 330 },
        hbmBandwidthGBs: 1008
      }]
    }]
  },
  {
    id: 'rtx-4080', name: 'NVIDIA RTX 4080', vendor: 'NVIDIA', family: 'Ada Lovelace',
    variants: [{
      id: 'sku', label: '16GB', hbmCapacityGB: 16,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 97.5, bf16: 97.5, fp8: 195, int8: 195 },
        hbmBandwidthGBs: 717
      }]
    }]
  },
  {
    id: 'rtx-pro-6000', name: 'NVIDIA RTX PRO 6000 Blackwell',
    vendor: 'NVIDIA', family: 'Blackwell',
    variants: [{
      id: 'workstation-96', label: 'Workstation 96GB', hbmCapacityGB: 96,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 252, bf16: 252, fp8: 504, int8: 504 },
        hbmBandwidthGBs: 1780
      }]
    }]
  },
  {
    id: 'b100', name: 'NVIDIA B100', vendor: 'NVIDIA', family: 'Blackwell',
    variants: [{
      id: 'sxm-192', label: 'SXM 192GB', hbmCapacityGB: 192,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 1750, bf16: 1750, fp8: 3500, int8: 3500 },
        hbmBandwidthGBs: 8000
      }]
    }]
  },
  {
    id: 'b200', name: 'NVIDIA B200', vendor: 'NVIDIA', family: 'Blackwell',
    // Per-GPU numbers derived from NVIDIA's HGX B200 spec table (8 GPUs):
    // 1.4 TB total memory ÷ 8 = 180 GB; FP16/BF16 36 PFLOPS sparse ÷ 8 ÷ 2 = 2250
    // TF dense; FP8 72 PFLOPS sparse ÷ 8 ÷ 2 = 4500 TF dense. FP4 dense (9000 TF
    // per GPU) is in the datasheet but not modeled here — the calc doesn't
    // currently support fp4 as a dtype.
    variants: [{
      id: 'sxm-180', label: 'SXM 180GB', hbmCapacityGB: 180,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 2250, bf16: 2250, fp8: 4500, int8: 4500 },
        hbmBandwidthGBs: 8000
      }]
    }]
  },
  {
    id: 'gb200', name: 'NVIDIA GB200', vendor: 'NVIDIA', family: 'Blackwell',
    // Per-GPU numbers derived from NVIDIA's GB200 Grace Blackwell Superchip
    // spec (1 Grace CPU + 2 Blackwell GPUs): 372 GB HBM3e ÷ 2 = 186 GB; 16 TB/s
    // ÷ 2 = 8 TB/s; 10 PFLOPS FP16/BF16 sparse ÷ 2 GPUs ÷ 2 (sparse→dense) =
    // 2500 TF dense; 20 PFLOPS FP8 sparse ÷ 2 ÷ 2 = 5000 TF dense; same for
    // INT8. GB200 runs at higher TDP than HGX B200, hence the ~11% compute
    // bump per GPU. FP4 not modeled (engine has no fp4 dtype).
    variants: [{
      id: 'nvl72-186', label: 'NVL72 (per GPU) 186GB', hbmCapacityGB: 186,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 2500, bf16: 2500, fp8: 5000, int8: 5000 },
        hbmBandwidthGBs: 8000
      }]
    }]
  },
  {
    id: 'mi300x', name: 'AMD Instinct MI300X', vendor: 'AMD', family: 'CDNA3',
    variants: [{
      id: 'oam-192', label: 'OAM 192GB', hbmCapacityGB: 192,
      operatingPoints: [
        {
          id: 'peak', label: 'Peak',
          tflops: { fp16: 1307, bf16: 1307, fp8: 2615, int8: 2615 },
          hbmBandwidthGBs: 5300
        },
        {
          id: 'achievable', label: 'Achievable',
          // AMD's own max-achievable FLOPs measurement (750W variant, ROCm 6.3.0,
          // hipBLASLt GEMM at 4096×4864×32896). Sustained clocks (1115-1230 MHz),
          // not boost — closer to what serving workloads actually deliver.
          // HBM not separately measured in this source; left at peak.
          tflops: { fp16: 654, bf16: 708, fp8: 1273 },
          hbmBandwidthGBs: 5300,
          tflopsSources: ['amd-rocm-mafs'],
          asOf: '2025-02',
          notes: 'hipBLASLt GEMM at sustained clocks, 750W; HBM not separately measured (using peak)'
        }
      ]
    }]
  },
  {
    id: 'mi325x', name: 'AMD Instinct MI325X', vendor: 'AMD', family: 'CDNA3',
    // Same CDNA3 silicon as MI300X, refreshed with 256GB HBM3e at 6 TB/s.
    // Compute is unchanged from MI300X per AMD's product page; only memory
    // capacity/bandwidth differ. No verified achievable-FLOPS source for
    // MI325X yet — MI300X's mamf measurement would not transfer cleanly
    // because the 256GB stack runs at different sustained clocks.
    variants: [{
      id: 'oam-256', label: 'OAM 256GB', hbmCapacityGB: 256,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 1300, bf16: 1300, fp8: 2610, int8: 2600 },
        hbmBandwidthGBs: 6000
      }]
    }]
  },

  // === Apple Silicon ===
  // Unified-memory architectures: capacity is shared with the OS and other workloads,
  // so usable headroom is materially lower than the figures below. No FP8/INT8/INT4
  // tensor acceleration on the GPU — those dtype keys are intentionally omitted; the
  // engine will throw "Operating point lacks tflops for ..." if a user picks them.
  // TFLOPS figures are best-effort estimates of GPU shader-core throughput (no tensor
  // cores). Cross-check before relying on absolute decode rates.
  {
    id: 'm3-pro', name: 'Apple M3 Pro', vendor: 'Apple', family: 'Apple Silicon',
    variants: [
      { id: '18gb', label: '18GB', hbmCapacityGB: 18,
        operatingPoints: [{ id: 'peak', label: 'Peak',
          tflops: { fp16: 14.8, bf16: 14.8 }, hbmBandwidthGBs: 150 }] },
      { id: '36gb', label: '36GB', hbmCapacityGB: 36,
        operatingPoints: [{ id: 'peak', label: 'Peak',
          tflops: { fp16: 14.8, bf16: 14.8 }, hbmBandwidthGBs: 150 }] }
    ]
  },
  {
    id: 'm3-ultra', name: 'Apple M3 Ultra', vendor: 'Apple', family: 'Apple Silicon',
    variants: [
      { id: '96gb', label: '96GB', hbmCapacityGB: 96,
        operatingPoints: [{ id: 'peak', label: 'Peak',
          tflops: { fp16: 114.7, bf16: 114.7 }, hbmBandwidthGBs: 820 }] },
      { id: '192gb', label: '192GB', hbmCapacityGB: 192,
        operatingPoints: [{ id: 'peak', label: 'Peak',
          tflops: { fp16: 114.7, bf16: 114.7 }, hbmBandwidthGBs: 820 }] },
      { id: '384gb', label: '384GB', hbmCapacityGB: 384,
        operatingPoints: [{ id: 'peak', label: 'Peak',
          tflops: { fp16: 114.7, bf16: 114.7 }, hbmBandwidthGBs: 820 }] }
    ]
  },
  {
    id: 'm4-pro', name: 'Apple M4 Pro', vendor: 'Apple', family: 'Apple Silicon',
    variants: [
      { id: '24gb', label: '24GB', hbmCapacityGB: 24,
        operatingPoints: [{ id: 'peak', label: 'Peak',
          tflops: { fp16: 18.4, bf16: 18.4 }, hbmBandwidthGBs: 273 }] },
      { id: '48gb', label: '48GB', hbmCapacityGB: 48,
        operatingPoints: [{ id: 'peak', label: 'Peak',
          tflops: { fp16: 18.4, bf16: 18.4 }, hbmBandwidthGBs: 273 }] }
    ]
  },
  {
    id: 'm4-max', name: 'Apple M4 Max', vendor: 'Apple', family: 'Apple Silicon',
    variants: [
      { id: '36gb', label: '36GB', hbmCapacityGB: 36,
        operatingPoints: [{ id: 'peak', label: 'Peak',
          tflops: { fp16: 34, bf16: 34 }, hbmBandwidthGBs: 546 }] },
      { id: '64gb', label: '64GB', hbmCapacityGB: 64,
        operatingPoints: [{ id: 'peak', label: 'Peak',
          tflops: { fp16: 34, bf16: 34 }, hbmBandwidthGBs: 546 }] },
      { id: '128gb', label: '128GB', hbmCapacityGB: 128,
        operatingPoints: [{ id: 'peak', label: 'Peak',
          tflops: { fp16: 34, bf16: 34 }, hbmBandwidthGBs: 546 }] }
    ]
  },
  {
    id: 'm5-pro', name: 'Apple M5 Pro', vendor: 'Apple', family: 'Apple Silicon',
    variants: [
      { id: '24gb', label: '24GB', hbmCapacityGB: 24,
        operatingPoints: [{ id: 'peak', label: 'Peak',
          tflops: { fp16: 22, bf16: 22 }, hbmBandwidthGBs: 310 }] },
      { id: '48gb', label: '48GB', hbmCapacityGB: 48,
        operatingPoints: [{ id: 'peak', label: 'Peak',
          tflops: { fp16: 22, bf16: 22 }, hbmBandwidthGBs: 310 }] }
    ]
  },
  {
    id: 'm5-max', name: 'Apple M5 Max', vendor: 'Apple', family: 'Apple Silicon',
    variants: [
      { id: '64gb', label: '64GB', hbmCapacityGB: 64,
        operatingPoints: [{ id: 'peak', label: 'Peak',
          tflops: { fp16: 70, bf16: 70 }, hbmBandwidthGBs: 640 }] },
      { id: '128gb', label: '128GB', hbmCapacityGB: 128,
        operatingPoints: [{ id: 'peak', label: 'Peak',
          tflops: { fp16: 70, bf16: 70 }, hbmBandwidthGBs: 640 }] }
    ]
  },

  // === Intel Gaudi ===
  // MME (matrix) throughput from Intel's Gaudi 3 white paper (Table 5 and the
  // Gaudi 2 vs Gaudi 3 comparison table). FP16 on Gaudi 3 MME is materially
  // slower than BF16 (459 vs 1678 TFLOPS) per the same table. INT8 is not in
  // the MME spec table; integer ops run on the TPC (vector) units which are
  // ~30× slower than the MME, so we omit int8 rather than blend the figures.
  {
    id: 'gaudi-2', name: 'Intel Gaudi 2', vendor: 'Intel', family: 'Gaudi 2',
    variants: [{
      id: 'oam-96', label: 'HL-225 OAM 96GB', hbmCapacityGB: 96,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { bf16: 432, fp8: 865 },
        hbmBandwidthGBs: 2460
      }]
    }]
  },
  {
    id: 'gaudi-3', name: 'Intel Gaudi 3', vendor: 'Intel', family: 'Gaudi 3',
    variants: [{
      id: 'oam-128', label: 'HL-325L OAM 128GB', hbmCapacityGB: 128,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 459, bf16: 1678, fp8: 1678 },
        hbmBandwidthGBs: 3700
      }]
    }]
  },

  // === Google TPU ===
  // Google's cloud docs report HBM bandwidth in GiB/s; we convert to decimal
  // GB/s here (×1024³/10⁹) to match the rest of the table. v5p lists FP8 at
  // the same throughput as BF16, suggesting no dedicated FP8 datapath — the
  // chip just runs FP8 at BF16 rate. v6e (Trillium) doesn't list FP8 at all;
  // INT8 doubles BF16, indicating a true INT8 datapath.
  {
    id: 'tpu-v5p', name: 'Google TPU v5p', vendor: 'Google', family: 'TPU',
    variants: [{
      id: 'chip', label: 'per chip 95GB', hbmCapacityGB: 95,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { bf16: 459, fp8: 459 },
        hbmBandwidthGBs: 2765
      }]
    }]
  },
  {
    id: 'tpu-trillium', name: 'Google TPU v6e (Trillium)', vendor: 'Google', family: 'TPU',
    variants: [{
      id: 'chip', label: 'per chip 32GB', hbmCapacityGB: 32,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { bf16: 918, int8: 1836 },
        hbmBandwidthGBs: 1759
      }]
    }]
  },

  // === AWS Neuron ===
  // AWS publishes Inferentia/Trainium specs at the instance (multi-chip) level;
  // per-chip values below are derived by division. Dense vs sparse is not
  // disclosed in AWS marketing — FP8 numbers here are taken at face value as
  // "peak" without sparsity, but treat with care.
  {
    id: 'trainium-2', name: 'AWS Trainium2', vendor: 'AWS', family: 'Neuron',
    // Per-chip derived from Trn2.48xlarge aggregates: 16 chips, 1.5 TB HBM3,
    // 46 TB/s aggregate bandwidth, 20.8 PFLOPS FP8 (no sparsity qualifier
    // given by AWS). BF16 not separately published; omitted rather than
    // guessed at half-FP8. The HBM rounds to ~94 GB exactly (1500/16) but AWS
    // datasheets typically refer to 96 GB per chip — keeping 96 to match.
    variants: [{
      id: 'chip', label: 'per chip 96GB', hbmCapacityGB: 96,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp8: 1300 },
        hbmBandwidthGBs: 2875
      }]
    }]
  },
  {
    id: 'inferentia-2', name: 'AWS Inferentia2', vendor: 'AWS', family: 'Neuron',
    // Per chip: 32 GB HBM (explicit on AWS Inf2 page); 9.8 TB/s aggregate ÷ 12
    // chips = ~817 GB/s per chip; 190 TFLOPS FP16 explicit on the AWS Neuron/
    // Inferentia product page. AWS lists FP8 support but no TFLOPS figure;
    // omitted rather than inferred.
    variants: [{
      id: 'chip', label: 'per chip 32GB', hbmCapacityGB: 32,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 190, bf16: 190 },
        hbmBandwidthGBs: 817
      }]
    }]
  },

  // === Cerebras Wafer-Scale ===
  // Wafer-scale architecture is a poor fit for the HBM-based roofline model
  // this calc assumes. The "capacity" field below is on-chip SRAM (44 GB),
  // not HBM — production deployments stream weights from external MemoryX
  // (1.5 TB / 12 TB / 1.2 PB tiers per Cerebras's CS-3 announcement), so
  // most LLM workloads don't actually live in the 44 GB shown here.
  // Bandwidth is on-die SRAM (21 PB/s); the 125 PFLOPS figure is Cerebras's
  // headline "AI compute" number with precision and sparsity unspecified in
  // the datasheet — treat the FP16/BF16 entries as a rough upper bound.
  {
    id: 'cerebras-wse3', name: 'Cerebras WSE-3', vendor: 'Cerebras', family: 'Wafer-Scale',
    variants: [{
      id: 'cs3', label: 'CS-3', hbmCapacityGB: 44,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 125000, bf16: 125000 },
        hbmBandwidthGBs: 21_000_000,
        notes: 'On-chip SRAM (44 GB) and SRAM bandwidth (21 PB/s), not HBM; weights normally stream from external MemoryX. 125 PFLOPS is Cerebras\'s headline figure — precision/sparsity not explicit in datasheet.'
      }]
    }]
  }
]
