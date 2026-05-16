import { mount } from 'svelte'
import App from './ui/App.svelte'
import { readUrlIntoStores, startUrlSync } from './ui/share'
import { initRouteSync } from './ui/route'

readUrlIntoStores()
const app = mount(App, { target: document.getElementById('app')! })
startUrlSync()
initRouteSync()
export default app
