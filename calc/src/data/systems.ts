import type { MultiAcceleratorSystem } from '../engine/types'

// Multi-GPU systems registry — concrete products users can buy or rent today
// (as of 2026-05). Each entry composes a GPU id + variant + count with a
// scale-up interconnect id, plus optional scale-out NIC info and cloud
// availability.
//
// Aggregate fields are denormalized for reviewer convenience and UI display
// only — not consumed by the engine. The engine looks up the underlying GPU
// and interconnect entries directly.
//
// "Available today" is loosely defined: shipping on OEM channels or in at
// least one cloud GA region. Pre-announced / preview-only systems
// (e.g. Rubin platform, Trainium3 instances) are omitted.

export const SYSTEMS: MultiAcceleratorSystem[] = [
  // === NVIDIA Hopper baseboards / nodes ===
  {
    id: 'hgx-h100-8',
    releaseDate: '2022-09',
    name: 'NVIDIA HGX H100 (8-GPU)',
    vendor: 'NVIDIA',
    generation: 'Hopper',
    formFactor: 'baseboard',
    accelerator: { id: 'h100', variantId: 'sxm-80', count: 8 },
    interconnectId: 'nvlink-4',
    scaleOutInterconnectId: 'ib-ndr',
    scaleOutNicsPerNode: 8,
    aggregate: {
      totalHbmGB: 640,
      fabricBidirectionalTBs: 7.2
    },
    availability: {
      onPrem: true,
      clouds: ['aws', 'azure', 'gcp', 'oci', 'coreweave', 'lambda', 'crusoe']
    },
    notes: 'Industry-standard 8× H100 SXM5 80GB baseboard. AWS P5.48xlarge, GCP A3, Azure ND H100 v5, OCI BM.GPU.H100.8. DGX H100 is NVIDIA-branded same baseboard + 2× Sapphire Rapids + 8× ConnectX-7 NDR.'
  },
  {
    id: 'hgx-h200-8',
    releaseDate: '2024-03',
    name: 'NVIDIA HGX H200 (8-GPU)',
    vendor: 'NVIDIA',
    generation: 'Hopper',
    formFactor: 'baseboard',
    accelerator: { id: 'h200', variantId: 'sxm-141', count: 8 },
    interconnectId: 'nvlink-4',
    scaleOutInterconnectId: 'ib-ndr',
    scaleOutNicsPerNode: 8,
    aggregate: {
      totalHbmGB: 1128,
      fabricBidirectionalTBs: 7.2
    },
    availability: {
      onPrem: true,
      clouds: ['aws', 'azure', 'gcp', 'oci', 'coreweave', 'lambda', 'crusoe']
    },
    notes: 'Drop-in H200 refresh on the H100 baseboard: same NVLink 4 / NVSwitch v3 fabric, larger 141GB HBM3e per GPU. AWS P5e.48xlarge, GCP A3 Ultra, Azure ND H200 v5.'
  },

  // === NVIDIA Blackwell ===
  {
    id: 'hgx-b200-8',
    releaseDate: '2025-01',
    name: 'NVIDIA HGX B200 (8-GPU)',
    vendor: 'NVIDIA',
    generation: 'Blackwell',
    formFactor: 'baseboard',
    accelerator: { id: 'b200', variantId: 'sxm-180', count: 8 },
    interconnectId: 'nvlink-5',
    scaleOutInterconnectId: 'ib-xdr',
    scaleOutNicsPerNode: 8,
    aggregate: {
      totalHbmGB: 1440,
      fabricBidirectionalTBs: 14.4
    },
    availability: {
      onPrem: true,
      clouds: ['aws', 'azure', 'gcp', 'coreweave', 'lambda', 'crusoe']
    },
    notes: '8× B200 SXM 180GB baseboard via NVSwitch v4. DGX B200 is the NVIDIA-branded variant with 2× Intel Emerald Rapids + 8× ConnectX-8 XDR. AWS P6-B200, GCP A4, Azure ND B200 v6.'
  },
  {
    id: 'gb200-nvl72',
    releaseDate: '2025-03',
    name: 'NVIDIA GB200 NVL72',
    vendor: 'NVIDIA',
    generation: 'Blackwell',
    formFactor: 'rack',
    accelerator: { id: 'gb200', variantId: 'nvl72-186', count: 72 },
    interconnectId: 'nvlink-5-nvl72',
    scaleOutInterconnectId: 'ib-xdr',
    scaleOutNicsPerNode: 72,  // 1 per GPU via ConnectX-8 on each compute tray
    aggregate: {
      totalHbmGB: 13392,
      fabricBidirectionalTBs: 129.6
    },
    availability: {
      onPrem: true,
      clouds: ['aws', 'gcp', 'coreweave', 'lambda', 'crusoe']
    },
    notes: 'Pre-integrated rack: 72 Blackwell GPUs + 36 Grace CPUs across 18 compute trays, NVLink Switch trays providing 130 TB/s of non-blocking fabric. AWS P6e-GB200, GCP A4X, CoreWeave, Crusoe.'
  },

  // === AMD ===
  {
    id: 'mi300x-8',
    releaseDate: '2023-12',
    name: 'AMD MI300X 8-OAM platform',
    vendor: 'AMD',
    generation: 'CDNA3',
    formFactor: 'baseboard',
    accelerator: { id: 'mi300x', variantId: 'oam-192', count: 8 },
    interconnectId: 'xgmi-4',
    scaleOutInterconnectId: 'ib-ndr',
    scaleOutNicsPerNode: 8,
    aggregate: {
      totalHbmGB: 1536,
      fabricBidirectionalTBs: 7.168
    },
    availability: {
      onPrem: true,
      clouds: ['azure', 'oci']
    },
    notes: '8× MI300X OAM in a fully-connected xGMI mesh (no switch). Supermicro AS-8125GS-TNMR2, Dell PowerEdge XE9680, HPE ProLiant XD685 are typical OEM hosts. Azure ND MI300X v5, OCI BM.GPU.MI300X.8.'
  },
  {
    id: 'mi325x-8',
    releaseDate: '2024-10',
    name: 'AMD MI325X 8-OAM platform',
    vendor: 'AMD',
    generation: 'CDNA3',
    formFactor: 'baseboard',
    accelerator: { id: 'mi325x', variantId: 'oam-256', count: 8 },
    interconnectId: 'xgmi-4',
    scaleOutInterconnectId: 'ib-ndr',
    scaleOutNicsPerNode: 8,
    aggregate: {
      totalHbmGB: 2048,
      fabricBidirectionalTBs: 7.168
    },
    availability: {
      onPrem: true,
      clouds: []
    },
    notes: 'MI325X refresh of the 8-OAM platform — same xGMI mesh, larger 256GB HBM3e per GPU. Limited cloud availability as of mid-2026; mostly OEM channel.'
  },

  // === Intel ===
  {
    id: 'gaudi3-hls',
    releaseDate: '2024-04',
    name: 'Intel Gaudi 3 HLS-3 (8-OAM)',
    vendor: 'Intel',
    generation: 'Gaudi 3',
    formFactor: 'baseboard',
    accelerator: { id: 'gaudi-3', variantId: 'oam-128', count: 8 },
    interconnectId: 'gaudi-3-roce',
    aggregate: {
      totalHbmGB: 1024,
      fabricBidirectionalTBs: 9.6
    },
    availability: {
      onPrem: true,
      clouds: ['intel-tiber']
    },
    notes: '8× Gaudi 3 OAM with integrated RoCE mesh — no external switch needed at 8-OAM scale. Supermicro and Dell ship HLS-3-compatible hosts. Intel Tiber Developer Cloud offers Gaudi 3 instances.'
  },

  // === Google TPU ===
  {
    id: 'tpu-v5p-8',
    releaseDate: '2023-12',
    name: 'Google TPU v5p single-host (8 chips)',
    vendor: 'Google',
    generation: 'v5p',
    formFactor: 'pod-slice',
    accelerator: { id: 'tpu-v5p', variantId: 'chip', count: 8 },
    interconnectId: 'tpu-ici-v5p',
    aggregate: {
      totalHbmGB: 760,
      fabricBidirectionalTBs: 9.6
    },
    availability: {
      onPrem: false,
      clouds: ['gcp']
    },
    notes: 'Smallest standalone TPU v5p slice — single host with 8 chips, 2×2×2 partial torus. Larger slices (up to 8960 chips per pod) scale the ICI domain proportionally with the same per-chip BW.'
  },
  {
    id: 'tpu-trillium-8',
    releaseDate: '2024-12',
    name: 'Google TPU v6e Trillium single-host (8 chips)',
    vendor: 'Google',
    generation: 'v6e',
    formFactor: 'pod-slice',
    accelerator: { id: 'tpu-trillium', variantId: 'chip', count: 8 },
    interconnectId: 'tpu-ici-trillium',
    aggregate: {
      totalHbmGB: 256,
      fabricBidirectionalTBs: 6.4
    },
    availability: {
      onPrem: false,
      clouds: ['gcp']
    },
    notes: 'Smallest standalone Trillium slice — single host, 8 chips. Pods scale to 256 chips in a 2D torus.'
  },

  // === AWS Neuron ===
  {
    id: 'aws-trn2-48xl',
    releaseDate: '2024-12',
    name: 'AWS Trn2.48xlarge',
    vendor: 'AWS',
    generation: 'Trainium2',
    formFactor: 'cloud-instance',
    accelerator: { id: 'trainium-2', variantId: 'chip', count: 16 },
    interconnectId: 'neuronlink-v2',
    scaleOutInterconnectId: 'aws-efa-v3',
    scaleOutNicsPerNode: 16,
    aggregate: {
      totalHbmGB: 1536,
      fabricBidirectionalTBs: 10.24
    },
    availability: {
      onPrem: false,
      clouds: ['aws']
    },
    notes: 'Single Trn2 instance: 16 Trainium2 chips on NeuronLink v2 scale-up fabric. EFA v3 NICs for multi-node training.'
  },
  {
    id: 'aws-trn2-ultraserver',
    releaseDate: '2024-12',
    name: 'AWS Trn2 UltraServer',
    vendor: 'AWS',
    generation: 'Trainium2',
    formFactor: 'cloud-instance',
    accelerator: { id: 'trainium-2', variantId: 'chip', count: 64 },
    interconnectId: 'neuronlink-v2',
    scaleOutInterconnectId: 'aws-efa-v3',
    aggregate: {
      totalHbmGB: 6144,
      fabricBidirectionalTBs: 40.96
    },
    availability: {
      onPrem: false,
      clouds: ['aws']
    },
    notes: '4× Trn2.48xlarge instances tied together as a single 64-chip NeuronLink domain. Aimed at 100B+ parameter training and large-model inference.'
  },
  {
    id: 'aws-inf2-48xl',
    releaseDate: '2023-04',
    name: 'AWS Inf2.48xlarge',
    vendor: 'AWS',
    generation: 'Inferentia2',
    formFactor: 'cloud-instance',
    accelerator: { id: 'inferentia-2', variantId: 'chip', count: 12 },
    interconnectId: 'pcie-gen5-x16',  // Inf2 uses PCIe for inter-chip, no dedicated scale-up fabric
    aggregate: {
      totalHbmGB: 384,
      fabricBidirectionalTBs: 0.768  // 12 × 64 GB/s PCIe Gen5 aggregate
    },
    availability: {
      onPrem: false,
      clouds: ['aws']
    },
    notes: 'Inference-focused: 12 Inferentia2 chips connected via PCIe (no dedicated scale-up like NeuronLink). For inference workloads where chips are partitioned across requests rather than sharing weights, PCIe headroom is rarely the bottleneck.'
  }
]
