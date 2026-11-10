# Annotation guidelines v3 (clean / offensive / hate)

**v3 (current):** profanity used only as an *intensifier or exclamation* is **clean** (decided after
the blind human annotation showed the author consistently labels "đẹp vl", "Đù anh Huấn" as clean,
and 52% of training comments containing "vl" were already labeled clean).
*Offensive* is now reserved for crude language that expresses anger, contempt or vulgarity.

Version 2 followed the **UIT-ViHSD** convention (Luu et al., 2021), which also underlies most of
the training data (ViHSD itself and the thesis pseudo-labels, produced by a ViHSD-trained model).
It replaces the thesis definitions, where *offensive* ("attacks an issue, organization or person")
and *hate* ("insults a group or someone") overlapped.

## The three labels

| Label | Definition | Key question |
|---|---|---|
| **clean** | No insult and no crude language. Criticism, complaints, sarcasm and negative opinions are clean when expressed without insults. **Profanity used only as an intensifier or exclamation is clean** ("đẹp vl", "hay vãi", "thổi giá vcl", "Đù", "Đậu má, 1m91"). | Is the only "bad word" an intensifier/exclamation? → clean |
| **offensive** | Crude or vulgar language that **expresses anger, frustration, contempt or vulgarity**, but **does not attack a specific target** (person, group, organization). Typical: curse phrases ("cc gì", "đéo", "mẹ nó", "dek có gì", "bỏ mẹ"), crude sexual content. | Is crude language used to vent or be vulgar, with no one targeted? → offensive |
| **hate** | **Attacks, insults, demeans or threatens a target**: an individual (including the person being replied to), a group (region, ethnicity, gender, religion, political group, fans…) or an organization. With or without profanity. | Is someone being insulted or demeaned? → Yes |

## Decision procedure

1. Does the comment insult, demean, threaten or dehumanize a **person, group or organization**
   (name-calling, slurs, wishing harm, comparing to animals)? → **hate**
2. Otherwise, does it use **curse phrases to vent anger/contempt** (đéo, đm, cc gì, mẹ nó, dek, bỏ mẹ…)
   or **crude sexual content**? → **offensive**
3. Otherwise → **clean** (including profanity used only as an intensifier or exclamation)

| Profanity type | Examples | Label (no target) |
|---|---|---|
| Intensifier after an adjective/verb | vl, vcl, vkl, vãi, vãi lồn, vler | clean |
| Exclamation | đù, má, đậu má, vãi | clean |
| Curse phrase / venting | đéo, éo, đm, cc gì, mẹ nó, bỏ mẹ, dek, cmn, địt | offensive |
| Any of the above aimed at a person/group | "thằng này ngu vl", "đm mày" | hate |

## Edge cases

| Case | Label | Example |
|---|---|---|
| Profanity as praise/intensifier | clean | "hay vãi", "đẹp vcl", "nhanh vl" |
| Curse phrase venting frustration, no target | offensive | "địt mẹ lại mất điện", "sút cc gì hiền vậy" |
| Polite but harsh criticism of a policy/organization | clean | "chính sách này làm dở quá" |
| Criticism of an organization **with insults** | hate | "công ty lừa đảo, bọn chó" |
| Insulting the person being replied to ("mày ngu") | hate | "mày im mẹ mồm đi", "não ngắn" |
| Regional/ethnic slurs (bắc kỳ chó, 3 que, tàu khựa, bò đỏ…) | hate | even without profanity |
| Mild teasing among friends, no vulgarity | clean | "ông này lầy thật" |
| Mild teasing with an intensifier, clearly friendly | clean | "ông này lầy vl" |
| Wishing death/violence on someone | hate | "con này nên phơi xác" |
| Mocking someone's body/appearance (body shaming), incl. their family | hate | "mẹ bạn béo", "béo như lợn" |
| Intensifier **aimed at a person's flaw** | hate | "m béo vl", "thằng này ngu vl" (vs. "đẹp vl" = clean) |
| Regional/ethnic slur on its own, even as a one-word comment | hate | "parky", "Đông Lào" |
| Neutral or positive use of a regional term | clean | "mấy anh Bắc Kỳ đẹp trai lắm" |
| Crude sexual content not aimed at a real target | offensive | |
| Sexual content about a real person | hate | |
| Only emoji, a name, a URL, unintelligible text | clean | |
| Quoting/reporting hate speech to criticize it | clean | "nó chửi 'bắc kỳ chó' là sai" |

## When unsure

- Between **offensive** and **hate**: ask *"Is there a target who is being put down?"*
  If yes → hate.
- Between **clean** and **offensive**: ask *"Is the bad word just making something stronger (clean), or is the
  writer venting/cursing (offensive)?"*
- Mark genuinely ambiguous items with `uncertain=1`; they are reviewed together afterwards.

## Annotation protocol for the 2026 test set

1. Claude pre-annotates all items following this document.
2. The author independently annotates a **random 100-item subset blind** (without seeing Claude's labels).
3. Agreement is measured with **Cohen's κ**; disagreements are discussed and the guideline is
   updated if a rule was unclear.
4. The final set is reported as *LLM-assisted annotation with human verification*.
