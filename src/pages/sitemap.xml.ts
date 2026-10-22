import { getPosts } from "../posts";
import { site } from "../site";

export async function GET() {
  const posts = await getPosts();
  const urls = [
    { loc: "/", lastmod: posts[0]?.data.date },
    ...posts.map(post => ({ loc: `/blog/${post.id}/`, lastmod: post.data.date })),
    { loc: "/blog/", lastmod: posts[0]?.data.date },
  ];
  const body = urls.map(({ loc, lastmod }) =>
    `<url><loc>${site.url}${loc}</loc>${lastmod ? `<lastmod>${lastmod}</lastmod>` : ""}</url>`).join("");
  return new Response(
    `<?xml version="1.0" encoding="utf-8" standalone="yes"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${body}</urlset>`,
    { headers: { "Content-Type": "application/xml" } });
}
