# Model proposal

Owner: Yuhan Zhang (Modelling lead)
Status: proposed (Oct 26, 2026), pending team agreement
Related: Issue #3, Issue #17 (literature reading)

Five models across three families, matching the course minimum (a trivial
baseline, a classical ML model, a deep learning model). Final choice:
Modelling lead.

## 1. Baselines (rule family)

- **Majority class**: always predicts the most frequent class. Reported as a
  reference line in every results table (it is a single number, not a trained
  model); it exposes class-imbalance artefacts such as high accuracy with
  near-zero macro-F1.
- **Keyword-rule baseline with statistically selected keywords**: after the
  split is frozen, the top-2 words per class are selected by mutual
  information computed on the **training split only**; each word is assigned
  to the class where its mutual information is highest (no duplicates). The
  rule counts keyword hits per class and predicts the most-voted class,
  falling back to majority on ties. No learned weights; k = 2 is fixed in
  advance and not tuned.

Rationale: rule-based floors are interpretable, the keyword selection is a
statistical procedure (mutual information) rather than hand-picking, and the
spec's baseline options explicitly include rule models.

## 2. Classical ML (linear family)

- **TF-IDF + logistic regression** (L2-regularized, class-weighted): linear
  models are strong baselines on bag-of-words text and often match deep
  models on short text (Joulin et al., EACL 2017); logistic regression is
  naturally calibrated, which feeds the reliability diagram required by the
  evaluation protocol, and its coefficients are interpretable.
- **TF-IDF + linear SVM**: the classical text classifier; its suitability for
  high-dimensional sparse text has been established for decades (Joachims,
  ECML 1998). Same features as the logistic regression, different decision
  rule, about a minute of CPU - shows whether the classifier family matters.

## 3. Deep learning

- **Fine-tuned ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`): BERT
  further pretrained on MIMIC-III clinical notes and PubMed/PMC full texts
  (Alsentzer et al., NAACL 2019 Clinical NLP Workshop), so it understands
  medical vocabulary and context. Fine-tuned on our notes with a
  classification head; expected to be the strongest model. The 512-token
  limit is handled by truncation (most notes fit under it).

Caveat supported by the literature: transformers are not guaranteed to beat
linear baselines on clinical text (Gao et al., IEEE JBHI 2021); keeping the
linear family makes that comparison explicit. Task-specific supervised
fine-tuning is also the recommended approach over LLM zero-shot prompting on
clinical reports (Sushil et al., JAMIA 2024), LLM workflows for specialty
referral itself still over- and under-triage (Gaber et al., npj Digital
Medicine 2025), and a recent MTSamples benchmark reports that earlier high
scores collapse once leakage is controlled (Barman & Barman, arXiv 2026) -
which is precisely our research question.

## Excluded: random forest

Trees underperform linear models on sparse high-dimensional text. Dropped to
keep the comparison clean; the empirical support for this will be confirmed
during the Issue #17 reading.

## Compute budget (Colab free tier, single T4 GPU)

| Model | Compute |
|---|---|
| Majority class | seconds, CPU |
| Keyword rule | seconds, CPU |
| TF-IDF + logistic regression | under a minute, CPU |
| TF-IDF + linear SVM | under a minute, CPU |
| ClinicalBERT fine-tuning | about 20-30 min per run (2,230 notes, 3 epochs) |

A few seeds of ClinicalBERT fit within roughly 2 hours; the whole comparison
is comfortably inside the free-tier budget.

## Planned leakage experiment

8.7% of notes contain their own label string in the text. Each model is run
on the original text and on a label-masked variant (label strings removed);
the score gap quantifies how much performance comes from leakage, which is
the research question itself.

## What the literature supports

The nine references support three parts of the plan, not only the model
choices:

| Paper | Model choice | Leakage / masking experiment | Evaluation protocol |
|---|---|---|---|
| Joachims 1998 | linear SVM | | |
| Joulin 2017 | linear baselines | | |
| Alsentzer 2019 | ClinicalBERT | | |
| Gao 2021 | transformer may not win | labels hinge on a few keywords; attention diluted - motivates masking | |
| Sushil 2024 | supervised fine-tuning route | | |
| Peluso 2024 | | | uncertainty and calibration reporting |
| Gaber 2025 | not LLM prompting (mis-triage) | | |
| Dahl 2025 | family comparison design | | |
| Barman 2026 | Bio_ClinicalBERT encoder | MTSamples scores were leakage-inflated - motivates our leakage-free benchmark | |

## Literature status

All citations below are real and verifiable, but the course rule requires
that a team member has actually read every cited paper. Reading is tracked in
Issue #17; the citations move from "candidate" to "cited" only after reading.
Four foundational classics support the method choices; three recent papers
(2024-2026) support the same claims with current evidence, including two
studies that used MTSamples itself.

- E. Alsentzer et al., "Publicly Available Clinical BERT Embeddings," in
  Proc. 2nd Clinical NLP Workshop (NAACL-HLT 2019), Minneapolis, MN, USA,
  2019, pp. 72-78, doi: 10.18653/v1/W19-1909. (foundational: ClinicalBERT
  original; domain pretraining helps clinical tasks but not all)
- T. Joachims, "Text Categorization with Support Vector Machines: Learning
  with Many Relevant Features," in Proc. ECML-98 (LNCS 1398), Chemnitz,
  Germany, 1998, pp. 137-142, doi: 10.1007/BFb0026683. (foundational: why
  linear SVMs suit high-dimensional sparse text)
- A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov, "Bag of Tricks for
  Efficient Text Classification," in Proc. 15th Conf. European Chapter of the
  ACL (EACL 2017), Valencia, Spain, 2017, pp. 427-431,
  doi: 10.18653/v1/E17-2068. (foundational: linear models are strong text
  baselines)
- S. Gao et al., "Limitations of Transformers on Clinical Text
  Classification," IEEE J. Biomed. Health Inform., vol. 25, no. 9,
  pp. 3596-3607, 2021, doi: 10.1109/JBHI.2021.3062322. (transformers did not
  beat simpler models on clinical text)
- M. Sushil et al., "A comparative study of large language model-based
  zero-shot inference and task-specific supervised classification of breast
  cancer pathology reports," J. Amer. Med. Inform. Assoc., vol. 31, no. 10,
  pp. 2315-2327, 2024, doi: 10.1093/jamia/ocae146. (recent: task-specific
  supervised fine-tuning versus LLM zero-shot on clinical reports)
- A. Peluso et al., "Deep learning uncertainty quantification for clinical
  text classification," J. Biomed. Inform., vol. 149, Art. no. 104576, 2024,
  doi: 10.1016/j.jbi.2024.104576. (recent: supports reporting uncertainty for
  clinical text classifiers, as our evaluation protocol does)
- S. Dahl, M. Bøgsted, T. Sagi, and C. Vesteghem, "Performance of Natural
  Language Processing for Information Extraction From Electronic Health
  Records Within Cancer: Systematic Review," JMIR Med. Inform., vol. 13,
  Art. no. e68707, 2025, doi: 10.2196/68707. (recent: systematic review
  comparing rule-based, traditional ML, CRF, neural and transformer model
  families on EHR text - supports our explicit family comparison design)
- F. Gaber et al., "Evaluating large language model workflows in clinical
  decision support for triage and referral and diagnosis," npj Digit. Med.,
  vol. 8, Art. no. 263, 2025, doi: 10.1038/s41746-025-01684-1. (recent:
  benchmark of LLM workflows for specialty referral from clinical notes on
  MIMIC; models over- and under-triage and struggle with numeric lab values -
  supports our supervised fine-tuning route over LLM prompting)
- P. K. Barman and P. K. Barman, "CLiGNet: Clinical Label-Interaction Graph
  Network for Medical Specialty Classification from Clinical Transcriptions,"
  arXiv:2603.22752, 2026, preprint, not peer-reviewed. (recent preprint on
  MTSamples itself: shows prior MTSamples results were inflated by SMOTE
  before splitting; their leakage-free benchmark uses Bio_ClinicalBERT as the
  encoder and reports macro-F1 0.279 on 40 classes - direct support for our
  research question and our leakage handling)
