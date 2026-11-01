import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { host: '127.0.0.1', port: 5173 },
  build: { rollupOptions: { output: { manualChunks(id) {
    if (id.includes('/node_modules/recharts/')) return 'charts'
    if (id.includes('/node_modules/lucide-react/')) return 'icons'
    if (id.includes('/node_modules/react/')) return 'react'
  } } } }
})
