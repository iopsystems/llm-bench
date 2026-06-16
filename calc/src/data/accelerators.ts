import type { AcceleratorSpec } from '../engine/types'

// Peak numbers from vendor datasheets. Dense compute (no sparsity).
// FP16/BF16 columns reflect Tensor Core peak.
//
// Entries span GPUs (NVIDIA/AMD), TPUs (Google), Trainium/Inferentia (AWS),
// Gaudi (Intel), Apple Silicon, and wafer-scale (Cerebras). The historic
// "tflops" + "hbmBandwidthGBs" + "hbmCapacityGB" vocabulary is GPU-rooted;
// for non-HBM parts (Apple unified memory, Cerebras on-die SRAM) the fields
// hold the equivalent measure, documented per-entry where it diverges.
export const ACCELERATORS: AcceleratorSpec[] = [
  {
    id: 'h100', name: 'NVIDIA H100', vendor: 'NVIDIA', family: 'Hopper',
    releaseDate: '2022-09', tier: 'datacenter',
    variants: [
      {
        id: 'sxm-80', label: 'SXM 80GB', hbmCapacityGB: 80,
        // 700W per nvidia.com/en-us/data-center/h100 ("Up to 700W (configurable)")
        powerCapW: 700,
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
        // 350W per NVIDIA H100 PCIe product brief
        powerCapW: 350,
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
        // 350W: low end of the H100 NVL configurable 350-400W range, used for
        // single-card PCIe deployments (NVIDIA H100 NVL product brief).
        powerCapW: 350,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 756, bf16: 756, fp8: 1513, int8: 1513 },
          hbmBandwidthGBs: 2400
        }]
      },
      {
        id: 'nvl-188', label: 'NVL (per GPU 94GB)', hbmCapacityGB: 94,
        // 400W per GPU: high end of NVL's 350-400W configurable range, the
        // setting at which the headline NVL TFLOPS figures land.
        powerCapW: 400,
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
    releaseDate: '2024-03', tier: 'datacenter',
    variants: [{
      id: 'sxm-141', label: 'SXM 141GB', hbmCapacityGB: 141,
      // 700W per nvidia.com/en-us/data-center/h200 ("Up to 700W (configurable)")
      powerCapW: 700,
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
    releaseDate: '2020-05', tier: 'datacenter',
    variants: [
      {
        id: 'sxm-40', label: 'SXM 40GB', hbmCapacityGB: 40,
        // 400W per NVIDIA A100 datasheet (SXM4 40GB)
        powerCapW: 400,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 312, bf16: 312, int8: 624 },
          hbmBandwidthGBs: 1555
        }]
      },
      {
        id: 'sxm-80', label: 'SXM 80GB', hbmCapacityGB: 80,
        // 400W per NVIDIA A100 datasheet; HGX A100 80GB CTS variants reach 500W
        // but those aren't the stock SXM SKU.
        powerCapW: 400,
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
        // 250W per NVIDIA A100 40GB PCIe product brief (PB-10137-001)
        powerCapW: 250,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 312, bf16: 312, int8: 624 },
          hbmBandwidthGBs: 1555
        }]
      },
      {
        id: 'pcie-80', label: 'PCIe 80GB', hbmCapacityGB: 80,
        // 300W per NVIDIA A100 80GB PCIe datasheet
        powerCapW: 300,
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
    id: 'l4', name: 'NVIDIA L4', vendor: 'NVIDIA', family: 'Ada Lovelace',
    releaseDate: '2023-05', tier: 'datacenter',
    variants: [{
      id: 'pcie-24', label: 'PCIe 24GB', hbmCapacityGB: 24,
      // 72W per nvidia.com/en-us/data-center/l4 ("Max thermal design power: 72W")
      powerCapW: 72,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        // NVIDIA's L4 product page quotes with-sparsity figures (242 / 485);
        // recording dense halves per the adding-hardware-sku skill.
        tflops: { fp16: 121, bf16: 121, fp8: 242.5, int8: 242.5 },
        hbmBandwidthGBs: 300
      }]
    }]
  },
  {
    id: 'l40s', name: 'NVIDIA L40S', vendor: 'NVIDIA', family: 'Ada Lovelace',
    releaseDate: '2023-08',
    // L40S is sold under nvidia.com/en-us/data-center/l40s and ships in OEM
    // 1U/2U inference nodes (Lenovo / Supermicro / Dell PowerEdge).
    tier: 'datacenter',
    variants: [{
      id: 'pcie-48', label: 'PCIe 48GB', hbmCapacityGB: 48,
      // 350W per nvidia.com/en-us/data-center/l40s
      powerCapW: 350,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 362, bf16: 362, fp8: 733, int8: 733 },
        hbmBandwidthGBs: 864
      }]
    }]
  },
  {
    id: 'rtx-5090', name: 'NVIDIA RTX 5090', vendor: 'NVIDIA', family: 'Blackwell',
    releaseDate: '2025-01', tier: 'consumer',
    variants: [{
      id: 'sku', label: '32GB', hbmCapacityGB: 32,
      // 575W per nvidia.com RTX 5090 spec page (Total Graphics Power)
      powerCapW: 575,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 209, bf16: 209, fp8: 419, int8: 419 },
        hbmBandwidthGBs: 1792
      }]
    }]
  },
  {
    id: 'rtx-5080', name: 'NVIDIA RTX 5080', vendor: 'NVIDIA', family: 'Blackwell',
    releaseDate: '2025-01', tier: 'consumer',
    variants: [{
      id: 'sku', label: '16GB', hbmCapacityGB: 16,
      // 360W per nvidia.com RTX 5080 spec page
      powerCapW: 360,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 112.6, bf16: 112.6, fp8: 225, int8: 225 },
        hbmBandwidthGBs: 960
      }]
    }]
  },
  {
    id: 'rtx-4090', name: 'NVIDIA RTX 4090', vendor: 'NVIDIA', family: 'Ada Lovelace',
    releaseDate: '2022-10', tier: 'consumer',
    variants: [{
      id: 'sku', label: '24GB', hbmCapacityGB: 24,
      // 450W per nvidia.com RTX 4090 spec page (Total Graphics Power)
      powerCapW: 450,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 165, bf16: 165, fp8: 330, int8: 330 },
        hbmBandwidthGBs: 1008
      }]
    }]
  },
  {
    id: 'rtx-4080', name: 'NVIDIA RTX 4080', vendor: 'NVIDIA', family: 'Ada Lovelace',
    releaseDate: '2022-11', tier: 'consumer',
    variants: [{
      id: 'sku', label: '16GB', hbmCapacityGB: 16,
      // 320W per nvidia.com RTX 4080 16GB spec page
      powerCapW: 320,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 97.5, bf16: 97.5, fp8: 195, int8: 195 },
        hbmBandwidthGBs: 717
      }]
    }]
  },
  {
    id: 'rtx-pro-6000', name: 'NVIDIA RTX PRO 6000 Blackwell',
    vendor: 'NVIDIA', family: 'Blackwell', releaseDate: '2025-03',
    tier: 'consumer',
    variants: [{
      id: 'workstation-96', label: 'Workstation 96GB', hbmCapacityGB: 96,
      // 600W per NVIDIA RTX PRO 6000 Workstation Edition spec page (Max-Q
      // variant is 300W and Server is also 600W; this entry is the full
      // Workstation Edition).
      powerCapW: 600,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 252, bf16: 252, fp8: 504, int8: 504 },
        hbmBandwidthGBs: 1780
      }]
    }]
  },
  {
    id: 'b100', name: 'NVIDIA B100', vendor: 'NVIDIA', family: 'Blackwell',
    releaseDate: '2025-01', tier: 'datacenter',
    variants: [{
      id: 'sxm-192', label: 'SXM 192GB', hbmCapacityGB: 192,
      // 700W per GPU — B100 was sized to drop into existing HGX H100 thermal
      // envelopes (700W air-cooled, per SemiAnalysis Blackwell deep dive).
      powerCapW: 700,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 1750, bf16: 1750, fp8: 3500, int8: 3500 },
        hbmBandwidthGBs: 8000
      }]
    }]
  },
  {
    id: 'b200', name: 'NVIDIA B200', vendor: 'NVIDIA', family: 'Blackwell',
    releaseDate: '2025-01', tier: 'datacenter',
    // Per-GPU numbers derived from NVIDIA's HGX B200 spec table (8 GPUs):
    // 1.4 TB total memory ÷ 8 = 180 GB; FP16/BF16 36 PFLOPS sparse ÷ 8 ÷ 2 = 2250
    // TF dense; FP8 72 PFLOPS sparse ÷ 8 ÷ 2 = 4500 TF dense. FP4 dense (9000 TF
    // per GPU) is in the datasheet but not modeled here — the calc doesn't
    // currently support fp4 as a dtype.
    variants: [{
      id: 'sxm-180', label: 'SXM 180GB', hbmCapacityGB: 180,
      // 1000W per GPU on HGX B200 (Lenovo ThinkSystem product guide, naming
      // the SKU "HGX B200 180GB 1000W GPU"; matches OEM datasheets across
      // Supermicro/Dell HGX B200 8-GPU boards).
      powerCapW: 1000,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 2250, bf16: 2250, fp8: 4500, int8: 4500 },
        hbmBandwidthGBs: 8000
      }]
    }]
  },
  {
    id: 'gb200', name: 'NVIDIA GB200', vendor: 'NVIDIA', family: 'Blackwell',
    releaseDate: '2025-03', tier: 'datacenter',
    // Per-GPU numbers derived from NVIDIA's GB200 Grace Blackwell Superchip
    // spec (1 Grace CPU + 2 Blackwell GPUs): 372 GB HBM3e ÷ 2 = 186 GB; 16 TB/s
    // ÷ 2 = 8 TB/s; 10 PFLOPS FP16/BF16 sparse ÷ 2 GPUs ÷ 2 (sparse→dense) =
    // 2500 TF dense; 20 PFLOPS FP8 sparse ÷ 2 ÷ 2 = 5000 TF dense; same for
    // INT8. GB200 runs at higher TDP than HGX B200, hence the ~11% compute
    // bump per GPU. FP4 not modeled (engine has no fp4 dtype).
    variants: [{
      id: 'nvl72-186', label: 'NVL72 (per GPU) 186GB', hbmCapacityGB: 186,
      // 1200W per GPU on the GB200 Grace Blackwell Superchip (liquid-cooled
      // NVL72 deployment). Each Superchip dissipates ~2700W = 2×1200W GPU +
      // ~300W Grace CPU & I/O. Source: Tweaktown / Wccftech reporting on
      // NVIDIA's GB200 product brief; matches the per-GPU thermal headroom
      // unlocked by NVL72 liquid cooling vs the 1000W air-cooled HGX B200.
      powerCapW: 1200,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 2500, bf16: 2500, fp8: 5000, int8: 5000 },
        hbmBandwidthGBs: 8000
      }]
    }]
  },
  {
    id: 'mi300x', name: 'AMD Instinct MI300X', vendor: 'AMD', family: 'CDNA3',
    releaseDate: '2023-12', tier: 'datacenter',
    variants: [{
      id: 'oam-192', label: 'OAM 192GB', hbmCapacityGB: 192,
      // 750W TBP per AMD MI300X datasheet (GD-176)
      powerCapW: 750,
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
    releaseDate: '2024-10', tier: 'datacenter',
    // Same CDNA3 silicon as MI300X, refreshed with 256GB HBM3e at 6 TB/s.
    // Compute is unchanged from MI300X per AMD's product page; only memory
    // capacity/bandwidth differ. No verified achievable-FLOPS source for
    // MI325X yet — MI300X's mamf measurement would not transfer cleanly
    // because the 256GB stack runs at different sustained clocks.
    variants: [{
      id: 'oam-256', label: 'OAM 256GB', hbmCapacityGB: 256,
      // 1000W TBP per AMD MI325X datasheet — power envelope bumped 250W over
      // MI300X to support the larger HBM3e stack at higher sustained clocks.
      powerCapW: 1000,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 1300, bf16: 1300, fp8: 2610, int8: 2600 },
        hbmBandwidthGBs: 6000
      }]
    }]
  },

  // === AMD Radeon (consumer RX + workstation PRO) ===
  // GDDR6, not HBM — the hbm* fields hold GDDR capacity/bandwidth (see header).
  // tflops.fp16/bf16 is the WMMA matrix (AI accelerator) rate, dense — the AMD
  // equivalent of NVIDIA Tensor Core peak used by the RTX consumer entries above,
  // NOT the vector/packed-FP16 shader rate. INT8 WMMA runs at 2× FP16 dense.
  // RDNA3 (Navi 3x) has no native FP8 WMMA → fp8 omitted. RDNA4 (Navi 4x) adds
  // FP8 WMMA and doubles per-CU matrix throughput: FP16 matrix dense = 4× FP32,
  // FP8/INT8 dense = 2× FP16. AMD's marketing TOPS are sparse (2×) — halved here.
  // Peak-only (no achievable tier): vendor-datasheet figures, matching the RTX rows.
  {
    id: 'rx-9070-xt', name: 'AMD Radeon RX 9070 XT', vendor: 'AMD', family: 'RDNA4',
    releaseDate: '2025-03', tier: 'consumer',
    variants: [{
      id: 'sku', label: '16GB', hbmCapacityGB: 16,
      // 304W TBP per AMD official spec / Wikipedia RDNA4 table
      powerCapW: 304,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        // FP32 48.7 → FP16 matrix dense 194.5; FP8/INT8 dense 389 (AMD quotes
        // 389 sparse FP16 / 779 sparse FP8/INT8).
        tflops: { fp16: 194.5, bf16: 194.5, fp8: 389, int8: 389 },
        hbmBandwidthGBs: 640
      }]
    }]
  },
  {
    id: 'rx-9070', name: 'AMD Radeon RX 9070', vendor: 'AMD', family: 'RDNA4',
    releaseDate: '2025-03', tier: 'consumer',
    variants: [{
      id: 'sku', label: '16GB', hbmCapacityGB: 16,
      // 220W TBP per AMD official spec (VideoCardz reporting of final specs)
      powerCapW: 220,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        // FP32 36.1 → FP16 matrix dense 144.4; FP8/INT8 dense 289.
        tflops: { fp16: 144.4, bf16: 144.4, fp8: 289, int8: 289 },
        hbmBandwidthGBs: 640
      }]
    }]
  },
  {
    id: 'radeon-ai-pro-r9700', name: 'AMD Radeon AI PRO R9700', vendor: 'AMD', family: 'RDNA4',
    releaseDate: '2025-07', tier: 'consumer',
    variants: [{
      id: 'sku', label: '32GB', hbmCapacityGB: 32,
      // 300W TDP per AMD AI PRO R9700 product brief (TechRadar/Phoronix)
      powerCapW: 300,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        // AMD datasheet: FP16 matrix 191 dense / 383 sparse; INT8 383 dense.
        tflops: { fp16: 191, bf16: 191, fp8: 383, int8: 383 },
        hbmBandwidthGBs: 644
      }]
    }]
  },
  {
    id: 'rx-7900-xtx', name: 'AMD Radeon RX 7900 XTX', vendor: 'AMD', family: 'RDNA3',
    releaseDate: '2022-12', tier: 'consumer',
    variants: [{
      id: 'sku', label: '24GB', hbmCapacityGB: 24,
      // 355W TBP per AMD spec / Wikipedia RDNA3 table
      powerCapW: 355,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        // AMD: 122.8 TFLOPS FP16 matrix (= 2× FP32 61.4). INT8 WMMA = 2× FP16.
        tflops: { fp16: 122.8, bf16: 122.8, int8: 245.6 },
        hbmBandwidthGBs: 960
      }]
    }]
  },
  {
    id: 'rx-7900-xt', name: 'AMD Radeon RX 7900 XT', vendor: 'AMD', family: 'RDNA3',
    releaseDate: '2022-12', tier: 'consumer',
    variants: [{
      id: 'sku', label: '20GB', hbmCapacityGB: 20,
      // 315W TBP per AMD spec / Wikipedia RDNA3 table
      powerCapW: 315,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        // FP32 51.5 → FP16 matrix 103; INT8 = 2× FP16.
        tflops: { fp16: 103, bf16: 103, int8: 206 },
        hbmBandwidthGBs: 800
      }]
    }]
  },
  {
    id: 'rx-7900-gre', name: 'AMD Radeon RX 7900 GRE', vendor: 'AMD', family: 'RDNA3',
    releaseDate: '2024-02', tier: 'consumer',
    variants: [{
      id: 'sku', label: '16GB', hbmCapacityGB: 16,
      // 260W TBP per Wikipedia RDNA3 table
      powerCapW: 260,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        // FP32 46.0 → FP16 matrix 92; INT8 = 2× FP16.
        tflops: { fp16: 92, bf16: 92, int8: 184 },
        hbmBandwidthGBs: 576
      }]
    }]
  },
  {
    id: 'rx-7800-xt', name: 'AMD Radeon RX 7800 XT', vendor: 'AMD', family: 'RDNA3',
    releaseDate: '2023-09', tier: 'consumer',
    variants: [{
      id: 'sku', label: '16GB', hbmCapacityGB: 16,
      // 263W TBP per Wikipedia RDNA3 table
      powerCapW: 263,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        // FP32 37.3 → FP16 matrix 74.6; INT8 = 2× FP16.
        tflops: { fp16: 74.6, bf16: 74.6, int8: 149.2 },
        hbmBandwidthGBs: 624
      }]
    }]
  },
  {
    id: 'rx-7700-xt', name: 'AMD Radeon RX 7700 XT', vendor: 'AMD', family: 'RDNA3',
    releaseDate: '2023-09', tier: 'consumer',
    variants: [{
      id: 'sku', label: '12GB', hbmCapacityGB: 12,
      // 245W TBP per Wikipedia RDNA3 table
      powerCapW: 245,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        // FP32 35.2 → FP16 matrix 70.3; INT8 = 2× FP16.
        tflops: { fp16: 70.3, bf16: 70.3, int8: 140.6 },
        hbmBandwidthGBs: 432
      }]
    }]
  },
  {
    id: 'radeon-pro-w7900', name: 'AMD Radeon PRO W7900', vendor: 'AMD', family: 'RDNA3',
    releaseDate: '2023-04', tier: 'consumer',
    variants: [{
      id: 'sku', label: '48GB', hbmCapacityGB: 48,
      // 295W TBP per AMD PRO W7900 datasheet (dual-slot variant)
      powerCapW: 295,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        // FP32 61.3 → FP16 matrix 122.6; INT8 = 2× FP16.
        tflops: { fp16: 122.6, bf16: 122.6, int8: 245.2 },
        hbmBandwidthGBs: 864
      }]
    }]
  },
  {
    id: 'radeon-pro-w7800', name: 'AMD Radeon PRO W7800', vendor: 'AMD', family: 'RDNA3',
    releaseDate: '2023-04', tier: 'consumer',
    variants: [{
      id: 'sku', label: '32GB', hbmCapacityGB: 32,
      // 260W TBP per AMD PRO W7800 datasheet
      powerCapW: 260,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        // FP32 45.2 → FP16 matrix 90.4; INT8 = 2× FP16.
        tflops: { fp16: 90.4, bf16: 90.4, int8: 180.8 },
        hbmBandwidthGBs: 576
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
    id: 'm3-pro', name: 'Apple M3 Pro', vendor: 'Apple', family: 'M3',
    releaseDate: '2023-10', tier: 'consumer',
    // powerCapW omitted across the Apple line: Apple doesn't publish a
    // per-SoC TDP comparable to a discrete GPU's TBP — the SoC integrates
    // CPU/GPU/NPU/memory under a single thermal envelope that varies by
    // chassis (MacBook Pro vs Mac Studio cooling differ materially).
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
    id: 'm3-ultra', name: 'Apple M3 Ultra', vendor: 'Apple', family: 'M3',
    releaseDate: '2025-03', tier: 'consumer',
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
    id: 'm4-pro', name: 'Apple M4 Pro', vendor: 'Apple', family: 'M4',
    releaseDate: '2024-10', tier: 'consumer',
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
    id: 'm4-max', name: 'Apple M4 Max', vendor: 'Apple', family: 'M4',
    releaseDate: '2024-10', tier: 'consumer',
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
    id: 'm5-pro', name: 'Apple M5 Pro', vendor: 'Apple', family: 'M5',
    releaseDate: '2026-01', tier: 'consumer',
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
    id: 'm5-max', name: 'Apple M5 Max', vendor: 'Apple', family: 'M5',
    releaseDate: '2026-01', tier: 'consumer',
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
    releaseDate: '2022-05', tier: 'datacenter',
    variants: [{
      id: 'oam-96', label: 'HL-225 OAM 96GB', hbmCapacityGB: 96,
      // 600W per Habana HL-225H / HL-225B mezzanine card datasheets
      powerCapW: 600,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { bf16: 432, fp8: 865 },
        hbmBandwidthGBs: 2460
      }]
    }]
  },
  {
    id: 'gaudi-3', name: 'Intel Gaudi 3', vendor: 'Intel', family: 'Gaudi 3',
    releaseDate: '2024-04', tier: 'datacenter',
    variants: [{
      id: 'oam-128', label: 'HL-325L OAM 128GB', hbmCapacityGB: 128,
      // 900W air-cooled per Intel Gaudi 3 product brief (HL-325L OAM
      // mezzanine card). Liquid-cooled variant goes to 1200W but isn't the
      // air-cooled SKU modeled here.
      powerCapW: 900,
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
    id: 'tpu-v5p', name: 'Google TPU v5p', vendor: 'Google', family: 'TPU v5',
    releaseDate: '2023-12', tier: 'datacenter',
    // powerCapW omitted: Google does not publish per-chip TDP for TPU v5/v6.
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
    id: 'tpu-trillium', name: 'Google TPU v6e (Trillium)', vendor: 'Google', family: 'TPU v6',
    releaseDate: '2024-12', tier: 'datacenter',
    // powerCapW omitted: Google does not publish per-chip TDP.
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
    id: 'trainium-2', name: 'AWS Trainium2', vendor: 'AWS', family: 'NeuronCore-v3',
    releaseDate: '2024-12', tier: 'datacenter',
    // powerCapW omitted: AWS does not officially publish per-chip TDP for
    // Trainium2. Third-party estimates (~500-700W) exist but aren't citable.
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
    id: 'inferentia-2', name: 'AWS Inferentia2', vendor: 'AWS', family: 'NeuronCore-v2',
    releaseDate: '2023-04', tier: 'datacenter',
    // powerCapW omitted: AWS does not publish per-chip TDP.
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
    id: 'cerebras-wse3', name: 'Cerebras WSE-3', vendor: 'Cerebras', family: 'WSE-3',
    releaseDate: '2024-03', tier: 'datacenter',
    variants: [{
      id: 'cs3', label: 'CS-3', hbmCapacityGB: 44,
      // powerCapW omitted: Cerebras quotes only system-level power (~23 kW
      // per CS-3) — no meaningful per-chip TDP since the WSE-3 *is* the system.
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 125000, bf16: 125000 },
        hbmBandwidthGBs: 21_000_000,
        notes: 'On-chip SRAM (44 GB) and SRAM bandwidth (21 PB/s), not HBM; weights normally stream from external MemoryX. 125 PFLOPS is Cerebras\'s headline figure — precision/sparsity not explicit in datasheet.'
      }]
    }]
  }
]
