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
- **TF-IDF + linear SVM**: the classical text classifier, same TF-IDF
  features as the logistic regression with a max-margin decision rule instead
  of a probability model; about a minute of CPU - shows whether the decision
  rule matters within the linear family.

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
fine-tuning is taken over LLM prompting for reproducibility (free Colab
budget, no closed API, results anyone can rerun), and a recent MTSamples
benchmark reports that earlier high scores collapse once leakage is
controlled (Barman & Barman, arXiv 2026) - which is precisely our research
question.

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

Five references support the whole plan, not only the model choices:

| Paper | Model choice | Leakage / masking experiment | Evaluation protocol |
|---|---|---|---|
| Alsentzer 2019 | ClinicalBERT | | |
| Joulin 2017 | linear baselines | | |
| Gao 2021 | transformer may not win | labels hinge on a few keywords; attention diluted - motivates masking | |
| Barman 2026 | Bio_ClinicalBERT encoder | MTSamples scores were leakage-inflated - motivates our leakage-free benchmark | |
| Peluso 2024 | | | uncertainty and calibration reporting |

## Literature status

All citations below are real and verifiable, but the course rule requires
that a team member has actually read every cited paper. Reading is tracked in
Issue #17; the citations move from "candidate" to "cited" only after reading.
Two foundational papers support the linear and deep model families, and three
recent papers (2021-2026) support the leakage experiment and the evaluation
protocol, including a study on MTSamples itself.

- E. Alsentzer et al., "Publicly Available Clinical BERT Embeddings," in
  Proc. 2nd Clinical NLP Workshop (NAACL-HLT 2019), Minneapolis, MN, USA,
  2019, pp. 72-78, doi: 10.18653/v1/W19-1909. (ClinicalBERT original; domain
  pretraining helps clinical tasks but not all)
- A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov, "Bag of Tricks for
  Efficient Text Classification," in Proc. 15th Conf. European Chapter of the
  ACL (EACL 2017), Valencia, Spain, 2017, pp. 427-431,
  doi: 10.18653/v1/E17-2068. (linear models are strong text baselines)
- S. Gao et al., "Limitations of Transformers on Clinical Text
  Classification," IEEE J. Biomed. Health Inform., vol. 25, no. 9,
  pp. 3596-3607, 2021, doi: 10.1109/JBHI.2021.3062322. (transformers did not
  beat simpler models on clinical text; labels hinge on few keywords)
- A. Peluso et al., "Deep learning uncertainty quantification for clinical
  text classification," J. Biomed. Inform., vol. 149, Art. no. 104576, 2024,
  doi: 10.1016/j.jbi.2024.104576. (uncertainty reporting for clinical text
  classifiers, as our evaluation protocol does)
- P. K. Barman and P. K. Barman, "CLiGNet: Clinical Label-Interaction Graph
  Network for Medical Specialty Classification from Clinical Transcriptions,"
  arXiv:2603.22752, 2026, preprint, not peer-reviewed. (MTSamples itself:
  prior high scores were inflated by SMOTE before splitting; leakage-free
  benchmark uses Bio_ClinicalBERT and reports macro-F1 0.279 on 40 classes -
  direct support for our research question and leakage handling)
