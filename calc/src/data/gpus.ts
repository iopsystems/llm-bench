import type { GpuSpec } from '../engine/types'

// Peak numbers from vendor datasheets. Dense compute (no sparsity).
// FP16/BF16 columns reflect Tensor Core peak.
export const GPUS: GpuSpec[] = [
  {
    id: 'h100', name: 'NVIDIA H100', vendor: 'NVIDIA', family: 'Hopper',
    variants: [
      {
        id: 'sxm-80', label: 'SXM 80GB', hbmCapacityGB: 80,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 989, bf16: 989, fp8: 1979, int8: 1979 },
          hbmBandwidthGBs: 3350
        }]
      },
      {
        id: 'pcie-80', label: 'PCIe 80GB', hbmCapacityGB: 80,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 756, bf16: 756, fp8: 1513, int8: 1513 },
          hbmBandwidthGBs: 2000
        }]
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
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 989, bf16: 989, fp8: 1979, int8: 1979 },
        hbmBandwidthGBs: 4800
      }]
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
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 312, bf16: 312, int8: 624 },
          hbmBandwidthGBs: 2039
        }]
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
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 312, bf16: 312, int8: 624 },
          hbmBandwidthGBs: 1935
        }]
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
    id: 'mi300x', name: 'AMD Instinct MI300X', vendor: 'AMD', family: 'CDNA3',
    variants: [{
      id: 'oam-192', label: 'OAM 192GB', hbmCapacityGB: 192,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 1307, bf16: 1307, fp8: 2615, int8: 2615 },
        hbmBandwidthGBs: 5300
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
  }
]
