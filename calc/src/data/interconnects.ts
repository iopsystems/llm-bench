import type { InterconnectSpec } from '../engine/types'

// Interconnect registry for multi-GPU topologies.
//
// Convention: `perGpuBandwidthGBs` is the aggregate BIDIRECTIONAL bandwidth
// per GPU/chip — the number vendors quote on spec sheets. For ring all-reduce
// math, callers need per-direction bandwidth (= half of bidirectional unless
// the entry states otherwise via `perDirectionGBs`).
//
// Coverage targets the scale-up interconnects that bind a server / superpod
// together. Scale-out fabrics (IB, RoCE, AWS EFA) are also included as
// reference points for the slower tier.
//
// PCIe rows are reference baselines; in practice a PCIe-only multi-GPU box
// also bounces through the CPU's root complex, which adds another factor of
// slowdown not modeled here.
//
// Achievable BW: entries may carry an optional `contention` (analytical
// fabric model) and/or `tiers` (empirical measurements). Both assume one
// workload owns the entire fabric — see the docstrings on FabricContention
// and InterconnectAchievableTier in src/engine/types.ts. None of the entries
// below populate these fields yet; population happens in a follow-up.
export const INTERCONNECTS: InterconnectSpec[] = [
  // === NVIDIA NVLink (per-GPU scale-up) ===
  {
    id: 'nvlink-3',
    name: 'NVLink 3',
    vendor: 'NVIDIA',
    generation: 'Gen3 (Ampere)',
    perGpuBandwidthGBs: 600,
    linksPerGpu: 12,
    perLinkGBs: 25,
    topology: 'switched',
    scale: 'intra-node',
    maxScaleUpGpus: 8,
    sources: ['nvidia-nvlink'],
    notes: 'A100 HGX baseboard via NVSwitch v2; 8-GPU non-blocking. Some A100 SXM4 configs also expose direct P2P NVLink between specific GPU pairs without NVSwitch.'
  },
  {
    id: 'nvlink-4',
    name: 'NVLink 4',
    vendor: 'NVIDIA',
    generation: 'Gen4 (Hopper)',
    perGpuBandwidthGBs: 900,
    linksPerGpu: 18,
    perLinkGBs: 25,
    topology: 'switched',
    scale: 'intra-node',
    maxScaleUpGpus: 8,
    sources: ['nvidia-nvlink'],
    notes: 'H100/H200 HGX baseboard via NVSwitch v3; 8 GPUs at 900 GB/s each, 7.2 TB/s aggregate switch bandwidth.'
  },
  {
    id: 'nvlink-4-nvl-256',
    name: 'NVLink 4 (DGX H100 SuperPOD)',
    vendor: 'NVIDIA',
    generation: 'Gen4 (Hopper)',
    perGpuBandwidthGBs: 900,
    linksPerGpu: 18,
    perLinkGBs: 25,
    topology: 'switched',
    scale: 'scale-up',
    maxScaleUpGpus: 256,
    sources: ['nvidia-nvlink'],
    notes: 'External NVLink Switch trays extend NVLink 4 to 256 GPUs; per-GPU bandwidth unchanged but the non-blocking domain is much larger than an HGX baseboard.'
  },
  {
    id: 'nvlink-5',
    name: 'NVLink 5',
    vendor: 'NVIDIA',
    generation: 'Gen5 (Blackwell)',
    perGpuBandwidthGBs: 1800,
    linksPerGpu: 18,
    perLinkGBs: 50,
    topology: 'switched',
    scale: 'intra-node',
    maxScaleUpGpus: 8,
    sources: ['nvidia-nvlink'],
    notes: 'HGX B200 / DGX B200 — 8-GPU baseboard via NVSwitch v4.'
  },
  {
    id: 'nvlink-5-nvl72',
    name: 'NVLink 5 (NVL72)',
    vendor: 'NVIDIA',
    generation: 'Gen5 (Blackwell)',
    perGpuBandwidthGBs: 1800,
    linksPerGpu: 18,
    perLinkGBs: 50,
    topology: 'switched',
    scale: 'scale-up',
    maxScaleUpGpus: 72,
    sources: ['nvidia-nvlink'],
    notes: 'GB200 NVL72 — 72 Blackwell GPUs in one NVLink switch domain. Aggregate switch bandwidth: 130 TB/s.'
  },

  // === AMD Infinity Fabric (xGMI) ===
  {
    id: 'xgmi-4',
    name: 'Infinity Fabric (xGMI 4)',
    vendor: 'AMD',
    generation: 'Gen4 (CDNA3)',
    perGpuBandwidthGBs: 896,
    linksPerGpu: 7,
    perLinkGBs: 64,
    topology: 'point-to-point',
    scale: 'intra-node',
    maxScaleUpGpus: 8,
    sources: ['amd-cdna3-whitepaper'],
    notes: 'MI300X / MI325X OAM platform: 7 xGMI links per GPU in a fully-connected 8-GPU mesh (no switch). 896 GB/s is the bidirectional aggregate widely cited by AMD; per-link figures derived. MI325X is silicon-equivalent to MI300X for IF purposes — not separately verified.'
  },

  // === Google TPU ICI ===
  {
    id: 'tpu-ici-v5p',
    name: 'TPU v5p ICI',
    vendor: 'Google',
    generation: 'v5p (3D torus)',
    perGpuBandwidthGBs: 1200,
    topology: '3d-torus',
    scale: 'scale-up',
    maxScaleUpGpus: 8960,
    sources: ['google-tpu-v5p-docs'],
    notes: 'Per-chip bidirectional ICI bandwidth; 8960 chips per pod with full 3D torus wrap-around at 4×4×4 slices and larger.'
  },
  {
    id: 'tpu-ici-trillium',
    name: 'TPU v6e ICI (Trillium)',
    vendor: 'Google',
    generation: 'v6e (2D torus)',
    perGpuBandwidthGBs: 800,
    linksPerGpu: 4,
    topology: '2d-torus',
    scale: 'scale-up',
    maxScaleUpGpus: 256,
    sources: ['google-tpu-v6e-docs'],
    notes: 'Per-chip bidirectional ICI bandwidth; 4 ICI ports per chip. Pod aggregate all-reduce bandwidth: 102.4 TB/s across 256 chips.'
  },

  // === Intel Gaudi ===
  {
    id: 'gaudi-2-roce',
    name: 'Gaudi 2 integrated RoCE',
    vendor: 'Intel',
    generation: 'Gaudi 2',
    perGpuBandwidthGBs: 600,
    linksPerGpu: 24,
    perLinkGBs: 12.5,
    topology: 'point-to-point',
    scale: 'intra-node',
    maxScaleUpGpus: 8,
    notes: '24 × 100 GbE on-die RoCE ports per accelerator; 21 used for scale-up in an 8-OAM box, remainder for scale-out. Bidirectional aggregate ~600 GB/s — Intel marketing figure, not separately verified. Architecture: GPU-driven Ethernet (no switch needed for the 8-OAM mesh).'
  },
  {
    id: 'gaudi-3-roce',
    name: 'Gaudi 3 integrated RoCE',
    vendor: 'Intel',
    generation: 'Gaudi 3',
    perGpuBandwidthGBs: 1200,
    linksPerGpu: 24,
    perLinkGBs: 25,
    topology: 'point-to-point',
    scale: 'intra-node',
    maxScaleUpGpus: 8,
    notes: '24 × 200 GbE on-die RoCE ports; 1.2 TB/s aggregate per accelerator. Same architectural pattern as Gaudi 2, doubled link speed. Numbers from Intel marketing; full Gaudi 3 white paper not separately consulted.'
  },

  // === AWS Neuron ===
  {
    id: 'neuronlink-v2',
    name: 'NeuronLink v2',
    vendor: 'AWS',
    generation: 'Trainium2',
    perGpuBandwidthGBs: 640,
    topology: 'point-to-point',
    scale: 'intra-node',
    maxScaleUpGpus: 16,
    notes: 'AWS-internal scale-up fabric within a Trn2.48xlarge (16 chips); per-chip bandwidth estimated from instance-level disclosures, AWS has not published authoritative per-chip NeuronLink numbers. UltraServer config (64 chips) extends the domain but per-chip BW is unverified.'
  },

  // === Apple UltraFusion ===
  {
    id: 'ultrafusion',
    name: 'Apple UltraFusion',
    vendor: 'Apple',
    generation: 'M-series Ultra',
    perGpuBandwidthGBs: 2500,
    topology: 'point-to-point',
    scale: 'die-to-die',
    maxScaleUpGpus: 2,
    notes: 'Two-die interposer fabric on M1/M2/M3 Ultra. 2.5 TB/s figure is from Apple\'s marketing — physical link only; this is on-package, not a true GPU-to-GPU interconnect in the cluster-scaling sense.'
  },

  // === PCIe (reference baseline) ===
  {
    id: 'pcie-gen4-x16',
    name: 'PCIe Gen4 x16',
    vendor: 'PCI-SIG',
    generation: 'Gen4',
    perGpuBandwidthGBs: 32,
    perDirectionGBs: 16,
    topology: 'point-to-point',
    scale: 'intra-node',
    notes: 'Per-direction = 31.5 GB/s effective after 128b/130b encoding. PCIe-only multi-GPU systems also hop through CPU root complex, which adds further latency not modeled.'
  },
  {
    id: 'pcie-gen5-x16',
    name: 'PCIe Gen5 x16',
    vendor: 'PCI-SIG',
    generation: 'Gen5',
    perGpuBandwidthGBs: 64,
    perDirectionGBs: 32,
    topology: 'point-to-point',
    scale: 'intra-node',
    notes: '63 GB/s effective per direction after encoding.'
  },
  {
    id: 'pcie-gen6-x16',
    name: 'PCIe Gen6 x16',
    vendor: 'PCI-SIG',
    generation: 'Gen6',
    perGpuBandwidthGBs: 128,
    perDirectionGBs: 64,
    topology: 'point-to-point',
    scale: 'intra-node',
    notes: 'PAM4 signalling; 121 GB/s effective per direction. Shipping in Blackwell-era hosts.'
  },

  // === InfiniBand / RoCE (scale-out) ===
  {
    id: 'ib-hdr',
    name: 'InfiniBand HDR',
    vendor: 'IBTA',
    generation: 'HDR (200 Gb/s)',
    perGpuBandwidthGBs: 50,
    perDirectionGBs: 25,
    topology: 'fat-tree',
    scale: 'scale-out'
  },
  {
    id: 'ib-ndr',
    name: 'InfiniBand NDR',
    vendor: 'IBTA',
    generation: 'NDR (400 Gb/s)',
    perGpuBandwidthGBs: 100,
    perDirectionGBs: 50,
    topology: 'fat-tree',
    scale: 'scale-out',
    notes: 'ConnectX-7 generation. DGX H100 nodes typically expose 8× ConnectX-7 = 400 GB/s aggregate scale-out per node.'
  },
  {
    id: 'ib-xdr',
    name: 'InfiniBand XDR',
    vendor: 'IBTA',
    generation: 'XDR (800 Gb/s)',
    perGpuBandwidthGBs: 200,
    perDirectionGBs: 100,
    topology: 'fat-tree',
    scale: 'scale-out',
    notes: 'ConnectX-8 / Quantum-X800; pairs with NVLink 5 era systems.'
  },
  {
    id: 'aws-efa-v3',
    name: 'AWS EFA v3',
    vendor: 'AWS',
    generation: 'EFAv3',
    perGpuBandwidthGBs: 100,
    perDirectionGBs: 50,
    topology: 'fat-tree',
    scale: 'scale-out',
    notes: 'Per-NIC scale-out for P5/Trn2 instances. SRD transport, not lossless IB; collective performance depends on Neuron / NCCL EFA plugin.'
  }
]
