# CLAUDE.md: EXSU 500 Team Project (Clinical NLP & LLMs)

McGill EXSU 500 Fundamentals of AI in Medicine, Fall 2026. Instructor: Dr. Amir Hooshiar.
Repo: https://github.com/gab-architech/exsu500-clinical-nlp (private for now)

The Team Project Specification on myCourses is the governing document. If anything here conflicts with it, the spec wins: flag the conflict, do not silently resolve it.

Personal preferences for individual team members live in each person's `CLAUDE.local.md` (git-ignored). This file holds team-wide rules only.

## Team

| Member | GitHub | Role |
|---|---|---|
| Gabriel Barrere | gab-architech | Data lead, team coordinator |
| Yuhan Zhang | yuhanniubility | Modelling lead (models TBD) |
| Jessica Oentoro | jessoen | Writing lead |
| Karman Kang | karman27 | Interface lead, writing support |

Every member still contributes code and writing; roles are accountability.

## Project goal

Define a healthcare AI problem on clinical/biomedical text, train and compare 2 to 3 model families under a defensible evaluation protocol, and report it as a short IEEE-style manuscript plus a 7-minute recorded video.

Guiding principle from the spec: novelty is NOT required. Competence, honesty and reproducibility are. A careful comparison of 3 well-understood models, correctly evaluated, beats an ambitious project with unclear evaluation. The protocol is graded as heavily as the numbers.

## Current status (update as decisions are made)

- Task statement: predict the medical specialty of a clinical note from its free text, as decision support for document routing and referral intake teams (human in the loop)
- Task type: single-label, multi-class text classification (11 classes; final list confirmed by manual validation)
- Dataset: MTSamples, Kaggle tboyle10/medicaltranscriptions (`mtsamples.csv`), CC0 as listed on Kaggle. Known issues (document-type labels, Surgery overlap, duplicates, label leakage in keywords/description/note text, rare classes) are listed in README.md
- Splits and metrics: split frozen Nov 3 (`splits/split_seed42.csv`: 70/15/15, stratified, seed 42, duplicates kept within one split); macro-F1 primary; training scope: Standard (#26)
- Midterm submitted (Group 5). Next deliverable: final, team target Fri Dec 4 (course deadline Dec 11). Timeline in README: validation runs done Nov 22, model lock and single `--final` test run Nov 25, manuscript draft Dec 1, video Dec 3.
- Branch/PR workflow: proposed, confirm with team (docs-only changes currently go straight to main)

## Deadlines (all 23:59 Eastern, submitted on myCourses)

| Date | Deliverable | Weight |
|---|---|---|
| Fri Oct 2 | Repo URL + initial README naming dataset, licence, problem, task type | required |
| Mon Nov 9 | Midterm Report: 3 pages IEEE two-column, >=15 refs, PDF + editable source | 20% |
| Fri Dec 11 | Final: manuscript 6 to 8 pages, 7-min video, repo URL, peer evaluation | 45% (+/-5 peer adj.) |
| Fri Dec 11 | Bonus: public repo (+3%), public deployed interface (+2%) | optional |

## HARD RULES (never violate)

### Data
- NEVER commit the dataset, raw or processed, to the repo. Provide download scripts/instructions in `data/README.md`.
- Dataset must be publicly downloadable with no credentialing, ethics approval, DUA or signed licence. MIMIC-type data via PhysioNet credentialing is NOT allowed.
- No MUHC/hospital data or identifiable PHI, ever.
- Must train on a free-tier cloud notebook (Google Colab, single GPU) within a few hours. Avoid huge archives.
- Record the dataset licence and cite it as its custodians request.

### Evaluation protocol (graded 35% of final)
- Declare train / validation / test splits BEFORE any modelling. Fix and save the split (indices or IDs) with a seed so it is reproducible.
- Save the split as `splits/split_seed42.csv` (row/group IDs + split column only, no text) and commit it. Every notebook loads this file instead of re-splitting.
- If data have grouping structure (e.g. several notes per patient), split at the GROUP level, never row level.
- Tune on validation only. The test set is touched ONCE, at the end. Never use test performance to choose models, hyperparameters, thresholds or features. If code would do this, stop and warn.
- The test set is loaded in exactly one notebook, the last one (e.g. `NN_final_test_evaluation.ipynb`). No other notebook reads test rows.
- Check for duplicates and near-duplicates across splits (leakage).
- Metrics must fit the task and class balance. For imbalanced classification report AUROC, AUPRC, and sensitivity and specificity at a stated threshold. Accuracy alone is not acceptable.
- Quantify variability: multiple random seeds or confidence intervals (e.g. bootstrap) on the primary metric. A single number without uncertainty is not a result.
- Produce at least one calibration or interpretability artifact: reliability curve, feature importance/SHAP, saliency, or confusion matrix with error analysis.

### Models (at least 2, 3 expected for text)
1. Trivial/statistical baseline (mandatory): majority class, simple rule, or plain logistic regression.
2. Classical ML: e.g. TF-IDF + logistic regression or linear SVM.
3. Modern deep learning: fine-tuned pretrained transformer (prefer a small or domain-specific model that fits Colab free tier; mind the 512-token limit).
Keep the comparison clean: 2 to 3 models done well beats 6 done loosely. Every model choice must be justifiable in the manuscript.

### Repository
- Never commit: data, model weights over ~100 MB (host externally), API keys, tokens, `.env` files. Respect `.gitignore`.
- Required by the final deadline: README (data access, how to reproduce, headline results table, interface URL if any), all code in followable run order, pinned environment file, short model card for the best model, MIT licence, CRediT statement, Use of GenAI statement.
- Notebooks/scripts are numbered in run order: `01_`, `02_`, ...
- Pin exact versions in `requirements.txt` once the environment is stable.
- Instructor access (adding @AHooshiar and TAs as collaborators) is handled by Gabriel manually. Do not do it.

### Academic integrity and GenAI
- NEVER invent citations, references, DOIs, results or numbers. Every reference must be real and must be read by a team member. If unsure a source exists, say so.
- Attribute any external code, figure or text (repos, tutorials, papers) at the point of use, in a comment.
- AI use is permitted but must be disclosed. After a work session where Claude made a substantive contribution, add one row to the Use of GenAI table in `docs/CONTRIBUTIONS.md`: tool + version (e.g. Claude Code, Claude Opus 5.5), date, what it was used for, how it was verified.
- Generative AI cannot be listed as an author or assigned CRediT roles.

### Interface (only if we attempt the +2% bonus)
- Must ALWAYS and prominently show: "Course prototype for research/educational purposes. Not a medical device. Not for clinical use."
- Must run our own trained model, show prediction + confidence + interpretability output, and handle invalid input without crashing.
- Must stay live with no login until at least May 1, 2027.

## Git workflow (shared repo)

- `main` is the shared branch and must always run. Pull before starting work and again before pushing.
- Claude asks before EVERY pull, commit and push. No exceptions, even mid-task.
- Before any commit, explain in plain language what changed and why, then show the commit message and wait for OK.
- Work on a short-lived branch per task (e.g. `gabriel/group-split`), open a pull request, merge into `main`. Tiny doc fixes can go straight to `main` if the team agrees.
- Commit messages follow Conventional Commits, lowercase, imperative, no trailing period:
  `feat: add stratified group split with fixed seed`, `fix: correct label mapping`, `docs: fill dataset table in readme`, `chore: pin requirements`.
- One logical change per commit. Small and frequent: commit history is graded per person.
- Commit only your own work from your own account. Never commit on behalf of a teammate.
- Never force push, never rewrite pushed history, never run reset/rebase on shared branches without asking. Before any destructive command (reset, force push, deleting files, rewriting history), stop and ask.
- On a merge conflict: stop, show both versions, and ask. Never resolve it by discarding a teammate's changes.
- GitHub access is via the `gh` CLI. Claude may open pull requests with `gh pr create`, after asking.

## Notebooks

- Clear all outputs before committing. Outputs can contain raw dataset text (a data rule violation), and they bloat diffs and cause merge conflicts.
- Figures meant for the manuscript or README are saved as files to `figures/` and committed, rather than kept as notebook outputs.
- One person edits a given notebook at a time (say so in the team chat). Notebook JSON merges badly.
- The first code cell detects Colab vs local and sets `ROOT` to the repo root. All paths are built from `ROOT`, never `../`.
- One `RANDOM_SEED = 42` in the CONFIG cell, passed to every split, model and sampler.
- When a function is needed in 2 or more notebooks (splits, metrics, bootstrap), move it to `src/` and import it.
- A notebook is done only when Restart & Run All passes top to bottom.

## Working conventions

- Prefer simple, readable code with comments over clever code. Every team member must be able to explain every line to the instructor.
- When there is a methodological choice (split strategy, metric, threshold, model), present 2 options with trade-offs and let the team member decide.
- Heavy training runs happen in Google Colab. Write code that runs in both Colab and locally.
- Style: no em dashes in any text written for the team or the repo.

## Repo layout

```
README.md              Project front page (Oct 2 version, grows to final)
CLAUDE.md              Team conventions for Claude Code
LICENSE                MIT
.gitignore             Blocks data, weights, secrets, CLAUDE.local.md
requirements.txt       Dependencies (pin exact versions later)
notebooks/             Numbered, run in order
  01_data_check.ipynb  Verifies dataset loads; class balance, lengths, duplicates, grouping
src/                   (to add) Shared functions used by 2+ notebooks
splits/                (to add) Saved split IDs, no text
figures/               (to add) Exported figures for manuscript/README
data/README.md         How to download the data (no data files)
docs/CONTRIBUTIONS.md  CRediT statement + Use of GenAI log
docs/model_card.md     (to add) Best model: intended use, data, metrics, limitations, excluded populations
```

## Midterm Report checklist (Nov 9, 3 pages, IEEE two-column)

Rubric: problem & significance 25%, literature review 35%, data & feasibility 25%, format 15%.
- [ ] Specific problem statement + clinical motivation (argued, not asserted)
- [ ] Literature review synthesized into an argument, >=15 references, mostly primary research from the last 5 years
- [ ] Quantitative dataset description (size, classes, distributions, known limitations/biases)
- [ ] Planned splits and metrics
- [ ] Candidate models + compute budget
- [ ] Work plan with roles and identified risks
- [ ] CRediT + Use of GenAI statements (outside page limit)

## Final manuscript checklist (Dec 11, 6 to 8 pages)

Structure: title/authors, abstract (<=250 words), introduction & clinical motivation, related work, data, methods, results, discussion incl. limitations, conclusion, CRediT, Use of GenAI, references.
- [ ] Model comparison table WITH uncertainty
- [ ] Calibration or interpretability artifact
- [ ] At least one original figure
- [ ] Honest statement of what the model cannot do and what would be needed before clinical use
- [ ] Contribution statement consistent with commit history
- [ ] Word file properties set (File > Properties): title = paper title, author = the four authors, not the template's "Paper Title" / "IEEE"

## Video (Dec 11, max 7:00, content after 7:00 is not marked)

All members on camera and introduce themselves. Cover problem, data, models, results + limitations, conclusion. Demo the interface if built, otherwise walk through results and key code.
