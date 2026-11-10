# Annotation guidelines v2 (clean / offensive / hate)

Version 2 follows the **UIT-ViHSD** convention (Luu et al., 2021), which also underlies most of
the training data (ViHSD itself and the thesis pseudo-labels, produced by a ViHSD-trained model).
It replaces the thesis definitions, where *offensive* ("attacks an issue, organization or person")
and *hate* ("insults a group or someone") overlapped.

## The three labels

| Label | Definition | Key question |
|---|---|---|
| **clean** | No offensive or vulgar language. Criticism, complaints, sarcasm and negative opinions are **clean** as long as they are expressed without insults or profanity. | Is there any vulgar word or insult? → No |
| **offensive** | Contains vulgar/profane words or crude language, but **does not attack a specific target** (person, group, organization). Profanity used as an intensifier, exclamation or joke. | Is the vulgar language aimed at someone? → No |
| **hate** | **Attacks, insults, demeans or threatens a target**: an individual (including the person being replied to), a group (region, ethnicity, gender, religion, political group, fans…) or an organization. With or without profanity. | Is someone being insulted or demeaned? → Yes |

## Decision procedure

1. Does the comment insult, demean, threaten or dehumanize a **person, group or organization**
   (name-calling, slurs, wishing harm, comparing to animals)? → **hate**
2. Otherwise, does it contain **vulgar/profane words** or crude sexual language? → **offensive**
3. Otherwise → **clean**

## Edge cases

| Case | Label | Example |
|---|---|---|
| Profanity as praise/intensifier | offensive | "đm hay vãi", "đẹp vcl" |
| Profanity as an exclamation of frustration, no target | offensive | "địt mẹ lại mất điện" |
| Polite but harsh criticism of a policy/organization | clean | "chính sách này làm dở quá" |
| Criticism of an organization **with insults** | hate | "công ty lừa đảo, bọn chó" |
| Insulting the person being replied to ("mày ngu") | hate | "mày im mẹ mồm đi", "não ngắn" |
| Regional/ethnic slurs (bắc kỳ chó, 3 que, tàu khựa, bò đỏ…) | hate | even without profanity |
| Mild teasing among friends, no vulgarity | clean | "ông này lầy thật" |
| Mild teasing with vulgarity, clearly friendly | offensive | "thằng lol này lầy vl" |
| Wishing death/violence on someone | hate | "con này nên phơi xác" |
| Crude sexual content not aimed at a real target | offensive | |
| Sexual content about a real person | hate | |
| Only emoji, a name, a URL, unintelligible text | clean | |
| Quoting/reporting hate speech to criticize it | clean | "nó chửi 'bắc kỳ chó' là sai" |

## When unsure

- Between **offensive** and **hate**: ask *"Is there a target who is being put down?"*
  If yes → hate.
- Between **clean** and **offensive**: ask *"Would this be bleeped on TV?"* If yes → offensive.
- Mark genuinely ambiguous items with `uncertain=1`; they are reviewed together afterwards.

## Annotation protocol for the 2026 test set

1. Claude pre-annotates all items following this document.
2. The author independently annotates a **random 100-item subset blind** (without seeing Claude's labels).
3. Agreement is measured with **Cohen's κ**; disagreements are discussed and the guideline is
   updated if a rule was unclear.
4. The final set is reported as *LLM-assisted annotation with human verification*.
