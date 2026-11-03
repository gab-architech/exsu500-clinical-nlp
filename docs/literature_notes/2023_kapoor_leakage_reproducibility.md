# Leakage and the reproducibility crisis in ML-based science

- **Citation:** S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," Patterns, vol. 4, no. 9, art. 100804, 2023. doi:10.1016/j.patter.2023.100804
- **Read by:** Gabriel Barrere, 2026-11-03
- **Peer-reviewed?** yes

**What they did:** compiled earlier reviews that found leakage in 329 papers across 17 fields, and grouped the errors into a taxonomy of 8 leakage types (e.g. no test set, preprocessing on train and test together, duplicates across splits, illegitimate features). Reproduced a set of civil war prediction papers and proposed "model info sheets" to document how leakage was ruled out.

**Main result:** in the civil war case study, every claim that complex ML beats logistic regression failed to reproduce because of leakage; once fixed, complex models did not do substantively better.

**Limitation:** the 329-paper count is aggregated from other reviews, not a systematic audit, so it is a lower bound and not a rate; the case study covers one field.

**Relevance to our project:** two of the 8 types apply directly: duplicates across splits (we deduplicate to unique texts) and illegitimate features (`keywords` contains the label in 97.9% of rows, so we model on `transcription` only). Also supports a fixed split and a single-use test set.
