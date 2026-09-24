/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        abyss: {
          950: "#050d1a",
          900: "#081527",
          850: "#0b1c33",
          800: "#0e2440",
          700: "#14345c",
        },
        sonar: {
          400: "#22d3ee",
          500: "#06b6d4",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
