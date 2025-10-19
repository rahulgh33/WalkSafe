import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: path.resolve(__dirname, "../web"), // write production build to repo/web
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      // proxy these endpoints to your Flask backend running on :5000
      "/route": "http://localhost:5000",
      "/route_map": "http://localhost:5000",
    },
  },
});