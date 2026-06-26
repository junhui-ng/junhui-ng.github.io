// astro.config.mjs
import { defineConfig, passthroughImageService } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkGithubAlerts from 'remark-github-alerts';

export default defineConfig({
  site: 'https://junhui-ng.github.io',
  integrations: [tailwind()],
  markdown: {
    remarkPlugins: [remarkMath, remarkGithubAlerts],
    rehypePlugins: [rehypeKatex],
  },
  image: {
    service: passthroughImageService()
  }
});