# Machine learning for medical imaging: methodological failures

- **Citation:** G. Varoquaux and V. Cheplygina, "Machine learning for medical imaging: methodological failures and recommendations for the future," npj Digital Medicine, vol. 5, art. 48, 2022. doi:10.1038/s41746-022-00592-y
- **Read by:** Gabriel Barrere, 2026-11-03
- **Peer-reviewed?** yes

**What they did:** reviewed how dataset choice, evaluation practice and publication incentives lead medical ML research astray, and gave recommendations for evaluation (adequate test sets, uncertainty reporting, strong baselines, avoiding test-set reuse).

**Main result:** small test sets give large uncertainty on reported performance, often larger than the differences between methods that papers claim, so many reported improvements are not meaningful.

**Limitation:** focused on medical imaging; the arguments transfer to clinical text, but its examples do not include NLP.

**Relevance to our project:** supports bootstrap confidence intervals on macro-F1 (our smallest class has about 8 test notes), comparing against simple baselines, and touching the test set once. We do not follow its call for external, multi-site validation: MTSamples is a single source.
