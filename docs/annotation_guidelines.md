# Annotation guidelines v3.2 (clean / offensive / hate)

**v3.2 (Nov 2026, after the blind check on test set 2): the hate / offensive boundary moved.**
The author's independent labels disagreed with v3.1 mainly on one pattern (10 of 24 disagreements):
insults aimed at one person with no reference to who they are ("thế thì m ngu", "thằng đần").
From v3.2 on:

| Label | v3.2 definition |
|---|---|
| **hate** | Attacks a person or group **because of who they are** (nationality, ethnicity, region, religion, gender, sexuality, age, political or fan-group slur such as "3 que", "ricon", "hàn xẻng"), **body shaming**, dehumanizing a group, sexual degradation, threats and death wishes. |
| **offensive** | Insults one person or the people being replied to **without** any of the above (intelligence, behaviour, character: "m ngu", "óc cu", "con điên này"), plus crude venting with no target (unchanged). |
| **clean** | Unchanged. Reminder from the same check: harsh words about what a wrongdoer did ("con này quá mất dạy, phải xử nghiêm") and plain-word generalizations ("bọn nó thượng đẳng") are clean. |

This matches the VLSP 2019 definition (hate targets individuals or groups *on the basis of their
characteristics*) and OLID's split between insults at an individual and at a group. Agreement with the
author on the 84 checked comments: Cohen's kappa 0.41 under v3.1 -> **0.70** under v3.2 (binary
toxic-vs-clean: 0.61 -> 0.84).

**Scope:** v3.2 is applied to **test set 2 only** (`labels/testset2_v2.csv`). The training data and the
first 2026 test set still follow v3.1, where an insult at one person is *hate*. Models trained on v3.1
labels are therefore expected to lose points on the hate/offensive split of test set 2, so the primary
metric there is **binary toxic vs clean**; the 3-class score is secondary. Rows below that say
"insulting one specific person ... = hate" describe v3.1.

Test set 2 also excludes English comments and comments shorter than 3 words (too little to judge).

---


**v3.1 (Nov 2026, after the second blind check on the active-learning batch):** criticism of a
person's or group's **behaviour or character**, without profanity, slurs, dehumanizing words or
mockery of appearance, is **clean** ("Thằng này ích kỷ, lười", "Toàn là diễn viên. Phét lác",
"những thành phần này phải tẩy chay"). Talking down to the person being replied to
("hùa câm mồm") stays **hate**; cursing to vent with no target ("đm phá bóng vl") stays **offensive**.

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
| Harsh criticism of a person's behaviour/character, no profanity or slur (v3.1) | clean | "thằng hề rất rẻ tiền", "trí thức rởm" |
| Fan nickname for a team/player | clean | "Mờ Ngu", "Xã đần" |
| Wishing legal punishment for criminals | clean | "tử hình hết bọn côn đồ" |
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
| Animal **idiom that criticizes behaviour**, not the person | clean | "làm mấy trò con bò", "nhảy dựng lên", "cày như trâu" |
| **Comparing the person** to an animal | hate | "sủa lên khác gì mấy con chó", "ngu như bò" |
| Slur word used in its **literal** sense (label by meaning, not by keyword) | clean | "mình có 3 que kem", "uống bò đỏ cho tỉnh ngủ" |
| Slur word where the sentence alone does not show which sense is meant | clean, `uncertain=1` | "bạn có 3 que không?" |
| Insulting one specific person's looks or thinking, no slur or threat (v3.1, confirmed) | hate | "tư duy của bạn chán vãi c", "xấu toàn diện thế này" |
| Teasing a celebrity's looks with a laugh, no demeaning words | clean, `uncertain=1` | "chị đẹp thế mà cưới a này xấu vl :)))" |
| Negative generalization about a country/group **with a demeaning word** (bọn, lũ, như lon, bọn tàu, xứ lừa…) | hate | "vn như lon về mọi mặt", "logic bọn nó là…" |
| Negative opinion about a country/government/group in plain words | clean | "kinh tế nước này yếu", "chính quyền làm chưa tốt" |
| Criticizing what a specific set of people did ("bọn lừa đảo", "bọn côn đồ đánh người") | clean | behaviour of wrongdoers, not a group identity |
| Crude sexual content not aimed at a real target | offensive | |
| Sexual content about a real person | hate | |
| Only emoji, a name, a URL, unintelligible text | clean | |
| Quoting/reporting hate speech to criticize it | clean | "nó chửi 'bắc kỳ chó' là sai" |

## Context-dependent comments (appearance, teasing)

The model (and the annotator) sees one comment without its thread. Label **hate only when the
comment itself shows intent to demean**: an animal/object comparison, mockery, an insult word, an
intensifier on a flaw, or a target's family. A plain remark whose offensiveness depends on who says it
to whom is labeled by its surface meaning (**clean**) and marked `uncertain=1`.

| Comment | Label | Why |
|---|---|---|
| "béo như lợn", "mẹ bạn béo", "m béo vl", "xấu như ma" | hate | demeaning on its own |
| "béo thế mà cũng đăng ảnh", "mập vậy ai thèm" | hate | clear mockery |
| "bạn mập quá đấy", "dạo này béo lên nhỉ" | clean, `uncertain=1` | depends on context |
| "mèo nhà mình béo quá 😍" | clean | no human target |

A context-aware model (parent comment + reply as input) is listed as future work.

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
