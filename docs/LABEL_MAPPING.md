# 🗺️ Label mapping

Every external dataset uses its own labels. This file records how each one is
translated into PinkFlag's 8 labels, and **why**. Every mapping is a judgment
call, so it's written down here and can be changed.

## MentalManip (ACL 2024)

Source: [huggingface.co/datasets/audreyeleven/MentalManip](https://huggingface.co/datasets/audreyeleven/MentalManip), CC BY-SA 4.0, consensus subset (2,915 dialogues).

| MentalManip technique | → PinkFlag | match | why |
|---|---|---|---|
| Denial | gaslighting | strong | denying what happened is the core of gaslighting |
| Feigning Innocence | gaslighting | strong | "I don't know what you're talking about" |
| Evasion | gaslighting | loose | dodging the topic distorts the conversation, but isn't always reality-denial |
| Rationalization | gaslighting | loose | justifying harm reframes reality; sometimes it's just an excuse |
| Playing Victim Role | guilt_tripping | strong | "poor me" to pressure you |
| Playing Servant Role | guilt_tripping | loose | "after all I do for you" sacrifice framing |
| Shaming or Belittlement | disrespect | strong | insults and putdowns |
| Intimidation | intimidation | strong | threats |
| Brandishing Anger | intimidation | strong | anger used to scare you into complying |
| Persuasion or Seduction | love_bombing | loose | charm used to get something; real love bombing is a narrower pattern |
| Accusation | *(unmapped)* | n/a | blame-shifting doesn't fit one label; candidate for a future label |
| *not manipulative* | green_flag | loose | "not manipulative" ≠ "healthy dating behaviour", but it's the closest we have |

**Notes**
- Labels are per **dialogue**, not per message. Rows have `unit = dialogue`.
- A dialogue can show several techniques. `labels` keeps all of them (`gaslighting|disrespect`); `label` is only filled when there's exactly one.
- `map_quality` records whether every technique mapped strongly. Experiments can train on `strong` rows only and compare.
- MentalManip has no **controlling** or **breadcrumbing** examples: those come from our own data.
- The language is from movie scripts, not texting. Our scraped and seed data cover Gen Z texting.
