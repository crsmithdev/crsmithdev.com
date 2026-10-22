import rss from "@astrojs/rss";
import { getPosts, summaryHtml } from "./posts";
import { site } from "./site";

export async function feed(title: string, description: string, path: string) {
  const posts = await getPosts();
  return rss({
    title,
    description,
    site: site.url + path,
    customData: "<language>en-us</language>",
    items: await Promise.all(posts.map(async post => ({
      title: post.data.title,
      link: `/blog/${post.id}/`,
      pubDate: new Date(post.data.date),
      description: await summaryHtml(post),
    }))),
  });
}
