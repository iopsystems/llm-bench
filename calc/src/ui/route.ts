// Hash-based view routing. Owns the path segment of location.hash; the
// calculator's shareable payload (share.ts) lives after `#calc?`. Kept
// separate so the two concerns don't fight over location.hash.
import { writable } from 'svelte/store'

export type Route =
  | { tab: 'calc' }
  | { tab: 'sim' }
  | { tab: 'info' }
  | { tab: 'info'; detail: { kind: 'model' | 'sku'; id: string } }

// Parse a raw location.hash (with or without leading '#') into a Route.
// Anything unrecognized falls back to the calculator.
export function parseRoute(hash: string): Route {
  const h = hash.replace(/^#/, '')
  if (h === '' || h === 'calc' || h.startsWith('calc?')) return { tab: 'calc' }
  if (h === 'sim'  || h.startsWith('sim?'))  return { tab: 'sim' }
  if (h === 'info') return { tab: 'info' }
  const m = h.match(/^info\/(model|sku)\/(.+)$/)
  if (m) return { tab: 'info', detail: { kind: m[1] as 'model' | 'sku', id: m[2] } }
  return { tab: 'calc' }
}

// Serialize a Route to a hash string (with leading '#'). For the calc and sim
// tabs an optional payload (the share.ts encodeState string) is appended as
// `?<payload>`. Both tabs use the same encoded payload (shared state); the
// hash prefix is what differentiates which tab the recipient lands on.
export function serializeRoute(route: Route, payload = ''): string {
  if (route.tab === 'calc') return payload ? `#calc?${payload}` : '#calc'
  if (route.tab === 'sim')  return payload ? `#sim?${payload}`  : '#sim'
  if ('detail' in route) return `#info/${route.detail.kind}/${route.detail.id}`
  return '#info'
}

// Live store of the current route, seeded from the URL and synced on hashchange.
export const route = writable<Route>(
  typeof window === 'undefined' ? { tab: 'calc' } : parseRoute(window.location.hash)
)

// Wire the hashchange listener. Call once at startup.
export function initRouteSync(): void {
  if (typeof window === 'undefined') return
  window.addEventListener('hashchange', () => route.set(parseRoute(window.location.hash)))
}

// Navigate by writing the hash; the hashchange handler updates the store.
export function navigate(next: Route): void {
  if (typeof window === 'undefined') return
  window.location.hash = serializeRoute(next)
}
