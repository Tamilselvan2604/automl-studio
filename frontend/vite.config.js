import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// GitHub Pages serves this repository at /automl-studio/; local dev stays at /.
const base = process.env.GITHUB_ACTIONS === 'true' ? '/automl-studio/' : '/';

export default defineConfig({
  plugins: [react()],
  base,
  server: { host: '0.0.0.0' },
});
