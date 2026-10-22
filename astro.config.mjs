import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";

export default defineConfig({
  site: "https://crsmith.dev",
  trailingSlash: "always",
  integrations: [mdx()],
  markdown: {
    // syntax.css picks the light or dark colours to follow the site's theme.
    shikiConfig: {
      themes: { light: "github-light-default", dark: "github-dark-default" },
      defaultColor: false,
    },
  },
});
