import { feed } from "../../feed";
import { site } from "../../site";

export const GET = () => feed(`Writing on ${site.title}`, `Recent content in Writing on ${site.title}`, "/blog/");
