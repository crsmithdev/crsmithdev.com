---
title: "Where the Koans Come From"
date: "2026-10-22T12:00:00-07:00"
draft: true
---

A monk asks Zhaozhou whether a dog has Buddha-nature, and Zhaozhou says *wu* (無), "no." That exchange opens the *Gateless Gate*, a collection of 48 koans compiled in 1228, and it may be the most famous single word in Zen. It's usually presented as timeless: a Tang dynasty master's spontaneous answer, handed down from teacher to student ever since.

The koans have a paper trail, though. The *Gateless Gate* and the *Blue Cliff Record* (1125) are both collections of older stories with commentary attached, and those stories had to come from somewhere. Most of the Chinese Buddhist canon is now digitized, so you can search all of it for the earliest text that tells each one. I did that for both collections, and for the *Platform Sutra*, a Zen text whose growth over five centuries happens to be unusually easy to measure.

<!--more-->

# The corpus

[CBETA](https://cbeta.org) publishes the Taishō canon and its supplements as TEI XML under [CC BY-NC-SA 4.0](https://cbeta.org/en/copyright). I used the Chan (Zen) and history volumes, plus the *Zutang ji* (祖堂集), a collection of Chan biographies from 952 that survived only as a Korean woodblock edition. That's 463 texts, and I could date about 170 of them from standard references. A date here means when a text was compiled or first printed, which isn't always when everything in it was written; that comes back to bite later, with the dog.

Classical Chinese has no spaces, so everything works on characters. To find a story in a text, I weight each five-character sequence in the story by how rare it is across the corpus. A rare phrase like Yunmen's 乾屎橛 ("dried turd") counts for a lot, and boilerplate like 僧問如何是 ("a monk asked, what is...") counts for almost nothing. Any text that holds enough of the weight gets a closer look: I align the story against it and measure how much of the story it holds, in order.

# How the Platform Sutra grew

The *Platform Sutra* tells how Huineng, an illiterate woodcutter, became the Sixth Patriarch of Chan. Two versions of it are in the canon: a manuscript found at Dunhuang, copied from an edition of about 780, and the edition Zongbao printed in 1291. Scholars have long known the sutra grew between the two. Aligning each of Zongbao's sentences against the Dunhuang text shows where.

Half of Zongbao's text is new, and almost none of the growth is in the sermons. Chapter 4, on samādhi and prajñā, is only 10% new. Huineng's encounters with his students are 86% new, and the imperial summons is 92% new. Several of the best-known scenes in Zen aren't in the Dunhuang text at all, like the monks arguing over whether the wind or the flag is moving, or Huineng's cry at the moment of his awakening: "Who would have thought that one's own nature is pure from the start?"

For about 40% of the new material, the earliest source in the canon is the *Jingde Record of the Transmission of the Lamp* (1004), the big Song dynasty genealogy of Chan masters, or the *Zutang ji* half a century before it. That includes the dialogues with Huairang and Xingsi, the two students every later Chan lineage traces itself back to. Half of it has no source before 1291 that I could find, probably because Zongbao worked from earlier editions that have since been lost.

Zongbao cut things too. The Dunhuang text ends with a list of the people who kept the sutra, each passing it to the next, and a warning not to give it to anyone unworthy. All of that is gone in 1291. The older text treats itself as a secret transmission; the newer one is a scripture for anyone.

# Tracing the koans

The koan collections were harder. Wumen, who compiled the *Gateless Gate*, rewords his stories, so a simple threshold on shared phrases found earlier versions for only 16 of his 48 cases.

A shuffle test got most of the rest. Someone retelling a story in their own words still keeps the order of events, and a stock phrase or a stray quotation doesn't. So for each candidate match I shuffle the story's characters five times, align the shuffles against the same text, and only accept the match if the real story aligns much better than any shuffle does. Short cases needed one more fix. A shuffled 15-character story matches a lot of single characters by chance, so in the end I only counted runs of two or more matching characters.

That traces 45 of the 48 *Gateless Gate* cases and 73 of the 100 *Blue Cliff* cases to an earlier text. I checked by hand every earliest match that the run-counting changed, and three of them were wrong: the same stock question as the case, with a different answer. Nothing I tried could tell those apart from a reworded story automatically, so I threw them out by hand.

For both collections, the median earliest attestation is 1050. Fourteen cases show up first in the *Zutang ji* (952), including Zhaozhou's "the cypress tree in the courtyard" and Mazu's "sun-face Buddha, moon-face Buddha." Most of the others show up first in early Song texts: the *Jingde Record*, the records of Fenyang (1024) and Yunmen (1076), and above all Baiyun Shouduan (died 1072), whose records hold 17 of the *Blue Cliff* cases before anyone else. The *Blue Cliff Record* is a commentary on a hundred cases picked by Xuedou, and these are the texts Xuedou and the generation after him would have been reading. As a set, the "ancient" koans look like an eleventh-century canon.

# The dog

Back to Zhaozhou's dog. Searching directly for the question and its answer turns up:

| Text | Date | Answer |
|---|---|---|
| *Wanling Record of Huangbo* | 857 | 無, with advice to watch the word day and night |
| *Record of Wuzu Fayan* | 1104 | 無 |
| *Record of Zhenxie Qingliao* | 1151 | yes and no |
| *Record of Hongzhi* | 1157 | yes and no |
| *Record of Dahui* | 1171 | the question twenty times, answered 無 |

The story isn't in the *Zutang ji* or the *Jingde Record*. If the Huangbo entry is right, the dog and its 無 were already in use as a meditation device in 857, two and a half centuries before anyone else mentions them.

I don't think it is. The passage sits in the last sermon of the *Wanling Record*, and the Taishō's own footnotes give it away. The Taishō prints a Ming dynasty edition and compares it against a Japanese print from 1673. In the 1673 print, the *Wanling Record* ends right before this sermon, with the line "here ends the *Wanling Record*," and the footnote says that print has nothing after that point. Two other collections that carry Huangbo's record, the *Gu zunsu yulu* and the *Zhiyue lu*, have the passage just before the sermon almost word for word, and neither has the sermon. The earliest dated text that does is a monastic anthology from 1313, which gives it as a separate address by Huangbo, not as part of his record. The sermon's vocabulary is a giveaway as well: "look at a case," "hold to the word 無," "become one piece" are the terms of the koan method of Dahui (1089–1163), and none of them appear anywhere else in Huangbo's two records.

So the dog's "no" first shows up with Wuzu Fayan around 1104, and a "yes" turns up in the Caodong records about fifty years later. Dahui, Wuzu's grandson in the lineage, repeats the "no" more than anyone, and with him it became the central koan of Zen practice, which is how it ended up at the front of the *Gateless Gate*. Sometime after that, someone gave the method a ninth-century pedigree by attaching it to Huangbo.

# Limits

"Earliest in the canon" isn't the same as "first told." These stories circulated orally, and plenty of texts are lost. About 290 of the texts I used have no date I trust, so none of them can count as earliest. A record can pick up later additions, the way Huangbo's did, and I've only tested that one. And because the method works on wording, a story retold in entirely different words won't be found.

Even so, the written record is clear about one thing: the koans Zen presents as Tang dynasty spontaneity were mostly collected and retold in the eleventh century, and canonized in the twelfth.
