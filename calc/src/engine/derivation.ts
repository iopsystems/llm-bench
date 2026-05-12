import type { DerivationStep } from './types'

export class DerivationBuilder {
  private readonly _steps: DerivationStep[] = []

  add(label: string, expression: string, value: number, unit: string): void {
    this._steps.push({ label, expression, value, unit })
  }

  steps(): DerivationStep[] {
    return [...this._steps]
  }
}
