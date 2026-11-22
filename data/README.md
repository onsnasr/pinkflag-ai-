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
| Web scraping (`python -m pinkflag.scrape`) | ✅ v0 | Beautiful Soup on hand-picked advice/psychology articles listed in [`configs/sources.yaml`](../configs/sources.yaml) + TikTok captions via the official oEmbed endpoint. Respects robots.txt, 2s delay, cached. First run: 415 raw quotes from 11 sites. **Unlabeled and noisy**: includes advice sentences and "how to respond" phrases that must be filtered during labeling. Output stays in `data/raw/` (not committed). |
| [MentalManip](https://aclanthology.org/2024.acl-long.206) (ACL 2024) | ✅ converted | `python -m pinkflag.datasets.mentalmanip` → 2,513 dialogues (2,217 single-label). Mapping decisions in [`docs/LABEL_MAPPING.md`](../docs/LABEL_MAPPING.md). | 4,000 multi-turn dialogues annotated for manipulation techniques + targeted vulnerabilities. From movie scripts, so formal language: we'll map its labels onto ours. |
| 🇹🇳 TSAC ([LGPL-3.0](https://github.com/fbougares/TSAC)) + TUNIZI ([CC BY 4.0](https://zenodo.org/records/4275240)) | ✅ converted | `python -m pinkflag.datasets.tunisian` → 85k Tunisian comments (92% Arabizi). **Sentiment labels, not red flags**: used to teach the model Derja. Usernames/phones/URLs masked. |
| 🇹🇳 Derja red-flag examples | ✍️ to write | [`seed/derja_examples.csv`](seed/derja_examples.csv), see [`docs/DERJA_GUIDE.md`](../docs/DERJA_GUIDE.md). Kept as the Derja test set. |
| Synthetic Gen Z conversations | 🔜 | Full conversations generated from patterns we see discussed on TikTok (we write down the *patterns*, we don't scrape TikTok). Every synthetic row gets human review. |
| Donated screenshots | 🔜 | Opt-in only, names/numbers blurred, consent recorded. This is the only set we'll trust as a **real-world test set**. |

## Rules
- No scraping. TikTok, Instagram and others forbid it in their terms.
- No real names, phone numbers or usernames in any committed file.
- Test sets come from a **different source** than training data. Otherwise we only measure how well the model copies our writing style.
- `data/raw/` and `data/interim/` are gitignored. Large datasets will be versioned with DVC, not git.
