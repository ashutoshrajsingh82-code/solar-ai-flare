/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        space: {
          950: '#060a12',
          900: '#0b1220',
          850: '#0f172a',
          800: '#15213b',
          750: '#1b2a4a',
          700: '#23365d',
          600: '#334b7a',
          500: '#48659d',
        },
        solar: {
          50: '#fffbeb',
          100: '#fef3c7',
          200: '#fde68a',
          300: '#fcd34d',
          400: '#fbbf24',
          500: '#f59e0b',
          600: '#d97706',
          700: '#b45309',
        },
        helios: {
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
        },
        solexs: {
          400: '#fb923c',
          500: '#f97316',
          600: '#ea580c',
        },
        plasma: {
          cyan: '#06b6d4',
          emerald: '#10b981',
          rose: '#f43f5e',
          violet: '#8b5cf6',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      boxShadow: {
        'glow-solar': '0 0 15px -3px rgba(245, 158, 11, 0.3)',
        'glow-helios': '0 0 15px -3px rgba(14, 165, 233, 0.3)',
        'glow-danger': '0 0 20px -2px rgba(239, 68, 68, 0.4)',
        'panel': '0 4px 20px -2px rgba(0, 0, 0, 0.5)',
      }
    },
  },
  plugins: [],
}
