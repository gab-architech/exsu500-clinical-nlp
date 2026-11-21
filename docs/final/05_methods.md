> Skeleton (Claude Code, Nov 21). Owner: TBD. Page budget: about 1.25 pages. Draft due Tue Dec 1.
> Sources: `docs/clinicalbert_pipeline.md`, `docs/results_validation.md` sections 1 to 3, `src/train_baselines.py`, `src/train_clinicalbert.py`, `src/evaluate.py`

# IV. Methods

- [ ] Five models and why each (baseline, classical, deep)
- [ ] Training details, the two validation-selected choices (LR C; ClinicalBERT epoch)
- [ ] Masking experiment (option C, 24 terms, 10.8% of notes)
- [ ] Metrics: macro-F1 primary, AUROC, AUPRC, sensitivity and specificity at 0.5, ECE; bootstrap CIs; seeds
- [ ] Test set read once after the Nov 25 lock (`results/test_runs.log`)
- [ ] Compute: Colab T4, about 2.5 min per ClinicalBERT run
