// Citation registry for operating-point provenance.
// Operating points reference these by key in their `sources` field.

export interface Source {
  title: string
  url: string
}

export const SOURCES = {
  'arxiv-2501-12084': {
    title: 'Dissecting the NVIDIA Hopper Architecture through Microbenchmarking',
    url: 'https://arxiv.org/abs/2501.12084'
  },
  'arxiv-2402-13499': {
    title: 'Benchmarking and Dissecting the NVIDIA Hopper GPU Architecture',
    url: 'https://arxiv.org/abs/2402.13499'
  },
  'arxiv-2510-27583': {
    title: 'AMD MI300X GPU Performance Analysis',
    url: 'https://arxiv.org/abs/2510.27583'
  },
  'arxiv-2512-02189': {
    title: "Microbenchmarking NVIDIA's Blackwell Architecture",
    url: 'https://arxiv.org/abs/2512.02189'
  },
  'arxiv-2502-05317': {
    title: 'Apple vs. Oranges: Evaluating Apple Silicon M-Series SoCs for HPC',
    url: 'https://arxiv.org/abs/2502.05317'
  },
  'mamf-finder': {
    title: 'stas00/ml-engineering — mamf-finder community table',
    url: 'https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks'
  },
  'nvbandwidth': {
    title: 'NVIDIA nvbandwidth',
    url: 'https://github.com/NVIDIA/nvbandwidth'
  },
  'amd-rocm-mafs': {
    title: 'AMD ROCm — Measuring Max-Achievable FLOPs',
    url: 'https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html'
  },
  'nvidia-cublas-12-0': {
    title: 'NVIDIA cuBLAS 12.0 Performance Blog',
    url: 'https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/'
  },
  // === Interconnect sources ===
  'nvidia-nvlink': {
    title: 'NVIDIA NVLink and NVLink Switch product page',
    url: 'https://www.nvidia.com/en-us/data-center/nvlink/'
  },
  'amd-cdna3-whitepaper': {
    title: 'AMD CDNA 3 Architecture White Paper',
    url: 'https://www.amd.com/system/files/documents/amd-cdna-3-white-paper.pdf'
  },
  'google-tpu-v5p-docs': {
    title: 'Google Cloud — TPU v5p system architecture',
    url: 'https://cloud.google.com/tpu/docs/v5p'
  },
  'google-tpu-v6e-docs': {
    title: 'Google Cloud — TPU v6e (Trillium) system architecture',
    url: 'https://cloud.google.com/tpu/docs/v6e'
  }
} as const satisfies Record<string, Source>

export type SourceKey = keyof typeof SOURCES
