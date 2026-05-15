import { mount } from 'svelte'
import App from './ui/App.svelte'
import { readUrlIntoStores, startUrlSync } from './ui/share'

// Restore any shared state from the URL hash before the app mounts so the
// initial render reflects the link, then start mirroring future store edits
// back to the URL.
readUrlIntoStores()
const app = mount(App, { target: document.getElementById('app')! })
startUrlSync()
export default app
