import { feed } from "../feed";
import { site } from "../site";

export const GET = () => feed(site.title, `Recent content on ${site.title}`, "/");
