# 🇹🇳 Writing Derja examples

PinkFlag should understand red flags in **Tunisian Derja**, not just English.
Public Tunisian datasets (TSAC, TUNIZI) teach the model *how Tunisians write*,
but none of them are about dating red flags. Those examples come from here.

## Where
[`data/seed/derja_examples.csv`](../data/seed/derja_examples.csv)

| column | what to put |
|---|---|
| `text` | the message, exactly how someone would text it |
| `label` | one of: `green_flag`, `love_bombing`, `gaslighting`, `controlling`, `guilt_tripping`, `disrespect`, `intimidation`, `breadcrumbing` |
| `script` | `latin` (Arabizi: "3lech"), `arabic` (علاش), or `mixed` |
| `source` | `handwritten_derja` |

## Goal
- **~12 per label → ~100 total**, green flags included (the model must learn what *fine* sounds like too)
- Mostly **Arabizi**, since that's how people really text; some Arabic script; some French mixed in
- Write it how people actually type: typos, "hhhh", emojis, no punctuation, all fine

## Rules
- ❌ No real names, phone numbers or usernames
- ❌ Don't copy real private messages. Write your own, inspired by what you've seen
- ✅ If you're unsure of a label, check the definitions in [`src/pinkflag/taxonomy.py`](../src/pinkflag/taxonomy.py)

## How these get used
1. **Test set (most important):** these are the only real Derja red-flag examples, so they're kept aside to measure honestly how well PinkFlag handles Derja.
2. **Translation (data augmentation):** English training examples get translated into Derja by an LLM, then reviewed by a native speaker. Those become *training* data.
