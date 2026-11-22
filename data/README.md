# 🗂️ Data

Every dataset is a CSV with three columns:

| column | meaning |
|---|---|
| `text` | the message |
| `label` | one key from [`src/pinkflag/taxonomy.py`](../src/pinkflag/taxonomy.py) |
| `source` | where it came from (`handwritten_v0`, `synthetic_v1`, `mentalmanip`, `donated`...) |

## Sources (current + planned)

| source | status | notes |
|---|---|---|
| `seed/seed_messages.csv` | ✅ v0 | ~185 hand-written example messages, including Gen Z slang. Fictional, good for bootstrapping, **not** for claiming real-world accuracy. |
| [MentalManip](https://aclanthology.org/2024.acl-long.206) (ACL 2024) | 🔜 | 4,000 multi-turn dialogues annotated for manipulation techniques + targeted vulnerabilities. From movie scripts, so formal language: we'll map its labels onto ours. |
| Synthetic Gen Z conversations | 🔜 | Full conversations generated from patterns we see discussed on TikTok (we write down the *patterns*, we don't scrape TikTok). Every synthetic row gets human review. |
| Donated screenshots | 🔜 | Opt-in only, names/numbers blurred, consent recorded. This is the only set we'll trust as a **real-world test set**. |

## Rules
- No scraping. TikTok, Instagram and others forbid it in their terms.
- No real names, phone numbers or usernames in any committed file.
- Test sets come from a **different source** than training data. Otherwise we only measure how well the model copies our writing style.
- `data/raw/` and `data/interim/` are gitignored. Large datasets will be versioned with DVC, not git.
