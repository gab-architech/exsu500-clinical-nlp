> AI-drafted (Claude Code, Nov 3). Owner: Yuhan Zhang to review and rewrite before Nov 4.
> Budget: about 0.5 page. From `docs/model_proposal.md` and Issue #3. Joachims, Joulin and Peluso (cited in the proposal) are not cited yet; Yuhan read them on Nov 5, so he can add them.
> Training scope: Standard, decided by the team (#26).
> Truncation strategy resolved Nov 5: measured 64.2% of notes exceed 512 tokens, so head+tail truncation (256+256) is used (#22).

# IV. Planned Methods

**Models.** We compare five models in three families, all trained on the same split.
(1) *Baselines:* a majority-class reference line, which exposes imbalance artefacts, and a keyword-rule model whose two keywords per class are selected by mutual information on the training set only and which predicts by keyword vote.
(2) *Linear models:* TF-IDF features with class-weighted logistic regression and with a linear SVM. Simple models matched transformers on full clinical documents [8], and logistic regression was as good as complex models once leakage was removed [15].
(3) *Transformer:* Bio_ClinicalBERT [7], fine-tuned with a classification head. The 512-token limit matters here: measured with its own tokenizer, the median note is 665 tokens and 64.2% exceed 512. We use head+tail truncation (first and last 256 tokens) so the opening and the impression are kept, and we report the share of truncated notes as a limitation [9]. We prefer supervised fine-tuning to LLM prompting because it is reproducible and fits the compute budget [10].

**Leakage experiment.** Each model is also run on a masked variant of the text with the label strings removed (8.7% of notes contain their own label); the score gap measures how much performance depends on this residual leakage.

**Evaluation.** Macro-F1 is the primary metric because it weights the 11 classes equally despite the 7.2x imbalance. We also report per-class precision, recall and F1, and one-vs-rest AUROC and AUPRC. Uncertainty follows the "Standard" scope: several random seeds for ClinicalBERT and 95% bootstrap confidence intervals on macro-F1, since small test sets can make differences between methods meaningless [16]. We will produce a confusion matrix with error analysis and a reliability (calibration) curve. Hyperparameters are tuned on the validation set only, and the test set is read once, in the final notebook. Test macro-F1 is reported with and without the 9 test notes that have a near-duplicate in another pile. For context we report the best leakage-free macro-F1 on MTSamples, 0.279 [5], while noting that it covers 40 raw labels and is not directly comparable to our 11 classes.

**Compute.** All models run on the Google Colab free tier (one T4 GPU). The baselines and linear models take under a minute on CPU; one ClinicalBERT run takes about 20 to 30 minutes (3 epochs), so several seeds fit in about two hours.
