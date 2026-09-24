import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev server proxies API calls to the FastAPI backend. Default :8000;
// override with VITE_API_PROXY_TARGET when running the backend elsewhere
// (e.g. http://127.0.0.1:8010). Production: `npm run build` then serve
// frontend/dist from the backend.
// FastAPI mounts the dist folder at /app, so the build base must be "/app/"
// — otherwise index.html references /assets/* at the origin root and the
// single-port demo serves a blank page. Dev keeps base "/" (server root).
export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: command === "build" ? "/app/" : "/",
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY_TARGET || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
}));
