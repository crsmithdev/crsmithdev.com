import { getCollection, type CollectionEntry } from "astro:content";
import { createSatteriMarkdownProcessor } from "@astrojs/markdown-satteri";

export type Post = CollectionEntry<"blog">;

// Like Hugo: production builds leave out drafts and posts dated in the future.
export async function getPosts(): Promise<Post[]> {
  const now = Date.now();
  const posts = await getCollection("blog", ({ data }) =>
    import.meta.env.DEV || (!data.draft && Date.parse(data.date) <= now));
  return posts.sort((a, b) => Date.parse(b.data.date) - Date.parse(a.data.date));
}

export const year = (post: Post) => post.data.date.slice(0, 4);

export const day = (post: Post) =>
  new Date(post.data.date.slice(0, 10) + "T00:00:00Z")
    .toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });

// The summary is the text before <!--more--> ({/* more */} in MDX), or else whole paragraphs up to 70 words, as in Hugo.
function summaryMarkdown(body: string): string {
  body = body.replace(/^import .*$/gm, "");
  const more = body.search(/<!--more-->|\{\/\* more \*\/\}/);
  if (more >= 0) return body.slice(0, more);
  const out: string[] = [];
  let words = 0;
  for (const para of body.trim().split(/\n\s*\n/)) {
    out.push(para);
    words += para.split(/\s+/).filter(Boolean).length;
    if (words >= 70) break;
  }
  return out.join("\n\n");
}

let processor: Awaited<ReturnType<typeof createSatteriMarkdownProcessor>> | undefined;

export async function summaryHtml(post: Post): Promise<string> {
  processor ??= await createSatteriMarkdownProcessor();
  return (await processor.render(summaryMarkdown(post.body ?? ""))).code.trim();
}

// Hugo counts the 160 characters before its minifier collapses the whitespace.
export async function description(post: Post): Promise<string> {
  let text = (await summaryHtml(post)).replace(/<[^>]*>/g, "");
  if (text.length > 160) text = text.slice(0, text.lastIndexOf(" ", 160) + 1) + "…";
  return text.replace(/\s+/g, " ").trim();
}
