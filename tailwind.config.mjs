/** @type {import('tailwindcss').Config} */
import typography from '@tailwindcss/typography'; // Import it

export default {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
	darkMode: 'class',
	theme: {
		extend: {
            colors: {
                stanford: { DEFAULT: '#8C1515', light: '#E86D7F' }
            }
        },
	},
	plugins: [
        typography()
    ],
}