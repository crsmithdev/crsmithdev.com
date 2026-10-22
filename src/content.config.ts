import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const blog = defineCollection({
  loader: glob({ pattern: ["*.{md,mdx}", "*/index.{md,mdx}"], base: "src/content/blog" }),
  schema: z.object({
    title: z.string(),
    // Kept as written so that the day shown is the day in the post's own offset.
    date: z.string(),
    draft: z.boolean().default(false),
    featuredImage: z.string().optional(),
  }),
});

const projects = defineCollection({
  loader: glob({ pattern: "*.md", base: "src/content/projects" }),
  schema: z.object({ title: z.string(), link: z.string(), weight: z.number() }),
});

const experience = defineCollection({
  loader: glob({ pattern: "*.md", base: "src/content/experience" }),
  schema: z.object({ title: z.string(), when: z.string(), small: z.boolean().default(false), weight: z.number() }),
});

const sections = defineCollection({
  loader: glob({ pattern: "*.md", base: "src/content/sections" }),
  schema: z.object({ title: z.string() }),
});

export const collections = { blog, projects, experience, sections };
