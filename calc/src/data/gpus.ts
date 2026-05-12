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
    id: 'mi300x', name: 'AMD Instinct MI300X', vendor: 'AMD', family: 'CDNA3',
    variants: [{
      id: 'oam-192', label: 'OAM 192GB', hbmCapacityGB: 192,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 1307, bf16: 1307, fp8: 2615, int8: 2615 },
        hbmBandwidthGBs: 5300
      }]
    }]
  }
]
