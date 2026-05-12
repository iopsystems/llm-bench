import { parseArgs } from 'node:util'
import { GPUS } from './data/gpus'
import { MODELS } from './data/models'
import { SOURCES, type Source } from './data/sources'
import { calculate } from './engine/calc'
import type { CalcInput, Dtype, Quantization, Workload } from './engine/types'

const DTYPES: readonly Dtype[] = ['fp32', 'fp16', 'bf16', 'fp8', 'int8', 'int4'] as const

function usage(): string {
  return `llm-calc — LLM Performance Calculator

USAGE
  llm-calc [OPTIONS]              Calculate performance metrics (default action)
  llm-calc list gpus              List available GPUs and their variants
  llm-calc list models            List available models with architecture info
  llm-calc --help                 Show this help

CALCULATE OPTIONS
  -g, --gpu <id>          GPU id (required). See 'list gpus'.
  -V, --variant <id>      GPU variant id. Defaults to first variant for the GPU.
  -m, --model <id>        Model id (required). See 'list models'.
  -p, --prompt <n>        Prompt tokens (default: 2048)
  -o, --output <n>        Output tokens (default: 512)
  -c, --concurrency <n>   Request concurrency (default: 1)
  -w, --weights <dtype>   Weight dtype (default: fp16)
  -k, --kv <dtype>        KV cache dtype (default: fp16)
  -a, --activations <dtype> Activation dtype (default: fp16)
      --format <json|table> Output format (default: json)

DTYPES: fp32, fp16, bf16, fp8, int8, int4

EXAMPLES
  llm-calc -g h100 -V sxm-80 -m llama-3.3-70b
  llm-calc -g h100 -m llama-3.3-70b --format table
  llm-calc list gpus
  llm-calc list models | grep qwen

OUTPUT
  JSON output is compact single-line, suitable for piping to jq.
  memory.fits=false means OOM; exit code is still 0.
`
}

function isDtype(s: string): s is Dtype {
  return (DTYPES as readonly string[]).includes(s)
}

function fail(msg: string, exitCode: number): never {
  process.stderr.write(`llm-calc: ${msg}\n`)
  process.exit(exitCode)
}

function runList(target: string | undefined): void {
  if (target === 'gpus') {
    for (const g of GPUS) {
      const variants = g.variants.map(v => v.id).join(', ')
      process.stdout.write(`${g.id.padEnd(20)} ${g.name.padEnd(40)} ${variants}\n`)
    }
    return
  }
  if (target === 'models') {
    for (const m of MODELS) {
      const params = (m.paramCount / 1e9).toFixed(2) + 'B'
      const gqa = `${m.numHeads}/${m.numKvHeads}`
      process.stdout.write(`${m.id.padEnd(24)} ${m.name.padEnd(28)} ${params.padStart(8)}  ${gqa}\n`)
    }
    return
  }
  fail(`unknown list target: ${target ?? '(missing)'}. Try 'gpus' or 'models'.`, 64)
}

function buildInput(values: Record<string, string | boolean | undefined>): CalcInput {
  const gpuId = (values['gpu'] as string | undefined) ?? fail('missing --gpu', 64)
  const gpu = GPUS.find(g => g.id === gpuId) ?? fail(`unknown gpu: ${gpuId}`, 1)
  const variantId = (values['variant'] as string | undefined) ?? gpu.variants[0].id
  if (!gpu.variants.find(v => v.id === variantId)) fail(`unknown variant for ${gpuId}: ${variantId}`, 1)
  const modelId = (values['model'] as string | undefined) ?? fail('missing --model', 64)
  const model = MODELS.find(m => m.id === modelId) ?? fail(`unknown model: ${modelId}`, 1)

  const weightsRaw = (values['weights'] as string | undefined) ?? 'fp16'
  const kvRaw = (values['kv'] as string | undefined) ?? 'fp16'
  const activationsRaw = (values['activations'] as string | undefined) ?? 'fp16'
  if (!isDtype(weightsRaw)) fail(`unknown dtype: ${weightsRaw}`, 1)
  if (!isDtype(kvRaw)) fail(`unknown dtype: ${kvRaw}`, 1)
  if (!isDtype(activationsRaw)) fail(`unknown dtype: ${activationsRaw}`, 1)
  const quant: Quantization = { weights: weightsRaw, kv: kvRaw, activations: activationsRaw }

  const workload: Workload = {
    promptTokens: Number((values['prompt'] as string | undefined) ?? 2048),
    outputTokens: Number((values['output'] as string | undefined) ?? 512),
    concurrency: Number((values['concurrency'] as string | undefined) ?? 1)
  }
  return { gpu, gpuVariantId: variantId, model, quant, workload }
}

function formatTable(input: CalcInput, result: ReturnType<typeof calculate>): string {
  const variant = input.gpu.variants.find(v => v.id === input.gpuVariantId)!
  const m = result.memory
  const GB = (n: number) => (n / 1024 ** 3).toFixed(2) + ' GB'
  const ms = (s: number) => (s * 1000).toFixed(2) + ' ms'
  const rate = (r: number) => r.toFixed(1) + ' tok/s'
  const fits = m.fits ? '✓ fits' : '✗ OOM'

  let out = ''
  out += `${input.gpu.name} ${variant.label}\n`
  out += `Model: ${input.model.name}\n`
  out += `Quant: weights=${input.quant.weights} kv=${input.quant.kv} act=${input.quant.activations}\n`
  out += `Workload: prompt=${input.workload.promptTokens} output=${input.workload.outputTokens} concurrency=${input.workload.concurrency}\n`
  out += `\n`
  out += `Memory: weights ${GB(m.weights)} | KV ${GB(m.kvCacheTotal)} | act ${GB(m.activationsPeak)} | total ${GB(m.total)} | ${fits}\n`
  const sameSet = (a?: string[], b?: string[]): boolean => {
    if (!a || !b) return false
    if (a.length !== b.length) return false
    const s = new Set(a)
    return b.every(k => s.has(k))
  }
  for (const [opId, perf] of Object.entries(result.perf)) {
    // De-dupe across the two axes for numbering.
    const order: string[] = []
    for (const k of perf.tflopsSources ?? []) if (!order.includes(k)) order.push(k)
    for (const k of perf.bandwidthSources ?? []) if (!order.includes(k)) order.push(k)
    const refs = order
      .map(k => ({ key: k, src: SOURCES[k as keyof typeof SOURCES] as Source | undefined }))
      .filter((x): x is { key: string; src: Source } => !!x.src)
      .map((x, i) => ({ key: x.key, n: i + 1, title: x.src.title, url: x.src.url }))
    const numOf = (k: string) => refs.find(r => r.key === k)?.n
    const allMarks = refs.map(r => `[${r.n}]`).join('')
    out += `[${opId}]${allMarks ? ' ' + allMarks : ''}\n`
    out += `  Prefill: TTFT ${ms(perf.ttftS)}  (${perf.prefill.regime}-bound)\n`
    out += `  Decode:  ${ms(perf.decode.timePerTokenS)}/tok  (${perf.decode.regime}-bound)\n`
    const perStream = 1 / perf.decode.timePerTokenS
    out += `  Rates:   in ${rate(perf.inputTokenRate)}, out ${rate(perStream)}/stream · ${rate(perf.outputTokenRate)} total\n`
    if (refs.length > 0) {
      const meta = [perf.asOf && `as of ${perf.asOf}`, perf.notes].filter(Boolean).join(' · ')
      if (meta) out += `  ${meta}\n`
      const merged = sameSet(perf.tflopsSources, perf.bandwidthSources)
      const fmtGroup = (label: string, keys: string[]) => {
        const marks = keys.map(k => numOf(k)).filter(n => n !== undefined).map(n => `[${n}]`).join('')
        return marks ? `${label}: ${marks}` : ''
      }
      const groups = merged
        ? [fmtGroup('Sources', perf.tflopsSources ?? [])]
        : [
            fmtGroup('TFLOPS', perf.tflopsSources ?? []),
            fmtGroup('Bandwidth', perf.bandwidthSources ?? [])
          ]
      const line = groups.filter(Boolean).join('  ')
      if (line) out += `  ${line}\n`
      for (const r of refs) out += `  [${r.n}] ${r.title} — ${r.url}\n`
    }
  }
  return out
}

function runCalc(values: Record<string, string | boolean | undefined>): void {
  const input = buildInput(values)
  let result
  try {
    result = calculate(input)
  } catch (err) {
    fail((err as Error).message, 1)
  }
  const format = (values['format'] as string | undefined) ?? 'json'
  if (format === 'json') {
    process.stdout.write(JSON.stringify(result) + '\n')
  } else if (format === 'table') {
    process.stdout.write(formatTable(input, result))
  } else {
    fail(`unknown format: ${format}. Use 'json' or 'table'.`, 64)
  }
}

function main(): void {
  const argv = process.argv.slice(2)
  if (argv.length === 0 || argv[0] === '--help' || argv[0] === '-h' || argv[0] === 'help') {
    process.stdout.write(usage())
    return
  }
  if (argv[0] === 'list') {
    runList(argv[1])
    return
  }
  let parsed
  try {
    parsed = parseArgs({
      args: argv,
      options: {
        gpu:         { type: 'string', short: 'g' },
        variant:     { type: 'string', short: 'V' },
        model:       { type: 'string', short: 'm' },
        prompt:      { type: 'string', short: 'p' },
        output:      { type: 'string', short: 'o' },
        concurrency: { type: 'string', short: 'c' },
        weights:     { type: 'string', short: 'w' },
        kv:          { type: 'string', short: 'k' },
        activations: { type: 'string', short: 'a' },
        format:      { type: 'string' },
        help:        { type: 'boolean', short: 'h' }
      },
      strict: true,
      allowPositionals: false
    })
  } catch (err) {
    fail((err as Error).message, 64)
  }
  if (parsed.values['help']) {
    process.stdout.write(usage())
    return
  }
  runCalc(parsed.values)
}

main()
