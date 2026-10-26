# Data cleaning and labelling

Author: Yuhan Zhang (Modelling lead)
Last updated: Oct 26, 2026
Input: `mtsamples.csv` (4,999 rows, 6 columns)

This document records every cleaning step, the count after each step, and the
reasoning behind each decision, so the pipeline can be reproduced and reported
in the manuscript. The numbers below come from running the pipeline on
`mtsamples.csv` downloaded from Kaggle (tboyle10/medicaltranscriptions).

## Pipeline at a glance

| Step | Action | Remaining | Change |
|---|---|---|---|
| 0 | Raw dataset | 4,999 rows | |
| 1 | Strip leading spaces in labels | 4,999 rows | 0 |
| 2 | Drop rows with no transcription | 4,966 rows | -33 |
| 3 | Drop document-type labels (6 labels) | 4,028 rows | -938 |
| 4 | Drop the Surgery label | 2,940 rows | -1,088 |
| 5 | Deduplicate to unique note texts | 2,279 texts | |
| 6 | Manual review of conflicting notes | 606 texts reviewed | |
| 7 | Map raw labels to 11 final classes | 2,230 notes | |

---

## Step 1: strip leading spaces in labels

**What:** every value in `medical_specialty` carried a hidden leading space
(for example `" Surgery"` instead of `"Surgery"`).

**Why:** the space broke equality checks, label mapping and any downstream
grouping. It was found during EDA when Surgery rows failed to match other
specialties. Stripping is lossless: no rows are removed.

**How:** `df["medical_specialty"].astype(str).str.strip()`.

## Step 2: drop rows with no transcription

**What:** drop the 33 rows whose `transcription` is empty.

**Why:** a model has nothing to read; there is no signal to classify.

## Step 3: drop document-type labels

**What:** drop every row whose label describes the document format rather than
a medical specialty:

| Label | Rows |
|---|---|
| Consult - History and Phy. | 516 |
| SOAP / Chart / Progress Notes | 166 |
| Discharge Summary | 108 |
| Emergency Room Reports | 75 |
| Office Notes | 51 |
| Letters | 23 |
| **Total** | **938** |

**Why:** the task is to route a note to a specialty team. Routing to
"Discharge Summary" or "Letters" is meaningless; these labels are document
types, not destinations. Decision agreed with the team (Issue #1).

## Step 4: drop the Surgery label

**What:** drop all 1,088 rows labelled `Surgery`.

**Why:** Surgery is a procedure type, not a specialty. Measured directly on the
data: 1,039 of the 1,088 rows are byte-for-byte identical to a note already
filed under a real specialty (for example one total hip replacement note exists
once as `Surgery` and once as `Orthopedic`, same 3,317 characters). The note
itself therefore survives under its real label; only the redundant copy is
removed.

The remaining 49 rows are surgical notes with no other filing. They were split:

- **21 notes** are general-surgery or wound-care procedures with no specialty
  home (wound debridement, incision and drainage, foreign body removal, burn
  escharotomy, dressing changes, central line placement). They were reviewed
  by the modelling lead and all 21 were confirmed as DROP.
- **28 notes** map to a real specialty (port insertions, breast biopsies,
  ERCP, hernia repair, appendectomy in pregnancy, etc.) and went to manual
  review (Step 6).

**How the split was made:** a deterministic keyword rule (terms such as
"debridement", "incision and drainage", "foreign body", "wound",
"escharotomy", "central venous") labelled each of the 49 notes as general
surgery or re-mappable. No LLM produced any label decision in this step or
anywhere else in the pipeline; the LLM assistant (Claude Code, disclosed in the
Use of GenAI log) only built the pipeline and the review tool. In summary:
byte-level duplicate matching removed 1,039 of the 1,088 Surgery rows; a
keyword rule split the remaining 49 into 28 re-mappable notes (human-reviewed
in the validation queue, Step 6) and 21 general-surgery notes (reviewed by the
modelling lead, all confirmed DROP).

## Step 5: deduplicate to unique texts

**What:** normalize every transcription (lowercase, strip punctuation,
collapse whitespace) and keep one example per unique text.

**Why:** 95.9% of the raw rows duplicate another row's text; most notes are
filed under several labels. The evaluation protocol requires the same note
text to stay inside a single split (no leakage across train/validation/test),
so a split is built at the text level: one text = one example = one label.
Collapsing first makes that guarantee explicit.

**Counts:** 2,940 rows collapse to **2,279 unique texts**:

- **578 texts** are filed under 2+ real specialties (label conflicts, for
  example one angiogram under both `Cardiovascular / Pulmonary` and
  `Radiology`); they go to manual review (Step 6).
- **1,701 texts** have a single label; they map directly in Step 7.

## Step 6: manual review of conflicting notes

**Queue:** 606 texts = 578 cross-label texts + 28 unique Surgery texts.

**Reviewers:** Jessica Oentoro and Karman Kang, two batches of 303 notes each,
in an offline browser tool (one note at a time; each note assigned one of the
11 final classes or marked DROP).

**First pass:** a keyword-rule suggestion (not an LLM) was shown with each note
as a starting point. The confidence labels were also rule-based, not model
probabilities: low = no keyword table matched (fallback General Medicine),
high = exactly one table matched, medium = two or more tables matched (the
first table in order won). Measured against the human labels the rule agreed
only 48.5% of the time (294/606): 62.1% on high-confidence suggestions,
54.8% on medium-confidence ones and 6.3% on low-confidence ones, so the
human review was essential rather than optional. The largest corrections
moved Radiology suggestions to Neurology (33), General Medicine suggestions
to Radiology (24) and Neurology (19), and Radiology suggestions to
Cardiovascular (17): the rule over-assigns Radiology for any imaging keyword
and uses General Medicine as a fallback when no table matches. The
suggestions must not be reported as model predictions in the manuscript;
they are a rule-based first pass. The full keyword tables and the
general-surgery terms are recorded in the notebook (Section 12.5).

**Result:** 603 notes assigned to a class and 3 marked DROP in the first
review. Three labels were then corrected after an audit by the LLM assistant
and confirmation by the modelling lead: two were found by reading the 28
unique-Surgery assignments (inguinal hernia repair General Medicine to
Gastroenterology; rectovaginal fistula Urology / Nephrology to Obstetrics /
Gynecology) and one by a keyword scan of the General Medicine class (buttock
abscess incision-and-drainage to DROP), leaving 602 assigned and 4 DROP.
Of the reviewed notes, 12 were assigned General Medicine in the first review
and 10 remain after the corrections; those 10 are non-surgical medical notes
(nine clearly medical, one burn-care follow-up).

**Inter-rater check:** only 11 notes were reviewed by both reviewers, agreeing
on 4 of them. The sample is too small to conclude anything; Gabriel's planned
blind double-review of a random 10% sample (Issue #14) will provide a proper
agreement measure (percent agreement and Cohen's kappa).

## Step 7: map raw labels to the 11 final classes

**Mapping:**

| Final class | Raw labels mapped in |
|---|---|
| Cardiovascular / Pulmonary | Cardiovascular / Pulmonary |
| Orthopedic | Orthopedic, Podiatry |
| Radiology | Radiology |
| General Medicine | General Medicine, Pain Management, Physical Medicine - Rehab, Sleep Medicine, Endocrinology, Rheumatology, Diets and Nutritions, Allergy / Immunology |
| Gastroenterology | Gastroenterology |
| Neurology | Neurology, Neurosurgery |
| Urology / Nephrology | Urology, Nephrology |
| Obstetrics / Gynecology / Pediatrics | Obstetrics / Gynecology, Pediatrics - Neonatal |
| ENT / Ophthalmology | ENT - Otolaryngology, Ophthalmology, Dentistry, Cosmetic / Plastic Surgery |
| Hematology - Oncology | Hematology - Oncology, Lab Medicine - Pathology |
| Psychiatry / Psychology | Psychiatry / Psychology |

**Why 11 classes:** the raw data has about 40 labels. Related specialties were
merged into families; Psychiatry / Psychology (50 notes) is kept as its own
class even though the target was 6-10, because it is a distinct specialty with
no natural merge target.

**Excluded labels** (no natural home; 65 texts): Autopsy, Bariatrics
(bariatric surgery pre/post-op records), Chiropractic, Dermatology,
Hospice - Palliative Care, IME-QME-Work Comp etc., Speech - Language, plus
the 21 general-surgery texts from Step 4 and 7 thyroid surgery texts dropped
at the mapping step (Step 7) as surgery with no routing class.

**Final dataset: 2,230 notes.**

| Class | Notes | Share |
|---|---:|---:|
| Cardiovascular / Pulmonary | 358 | 16.1% |
| Orthopedic | 345 | 15.5% |
| General Medicine | 286 | 12.8% |
| ENT / Ophthalmology | 221 | 9.9% |
| Gastroenterology | 204 | 9.1% |
| Urology / Nephrology | 198 | 8.9% |
| Neurology | 194 | 8.7% |
| Obstetrics / Gynecology / Pediatrics | 174 | 7.8% |
| Radiology | 105 | 4.7% |
| Hematology - Oncology | 95 | 4.3% |
| Psychiatry / Psychology | 50 | 2.2% |
| **Total** | **2,230** | 100% |

Class imbalance is 7.2x (358 vs 50). The smallest class (Psychiatry /
Psychology, 50 notes) yields roughly 30 train / 10 validation / 10 test under a
60/20/20 split, so its metrics will be noisy; this is handled with macro-F1,
bootstrap confidence intervals and class weighting rather than by changing the
class list.

## Input features (leakage control)

Modelling uses the `transcription` text only. Three leakage sources were
measured in EDA:

- `keywords` contains the label itself in 97.9% of the rows that have
  keywords (3,850 of 3,931 rows; 77% of all rows), and `description`
  contains it in 1.0% of rows. Both columns are excluded from the modeling
  input.
- the label string also appears inside the `transcription` itself in 8.7% of
  rows (for example "referral to Urology" or "Pediatric Urology Clinic").
  MTSamples transcriptions are single-line, so header-line stripping is not
  applicable: it would delete the whole note. In-text leakage is handled at
  modeling time, where a masked variant of the text (label strings removed)
  is planned so the effect of this residual leakage can be quantified, which
  is exactly the research question.

## Reproducibility and file policy

- No note text is committed to the repo (course rule). The validation CSVs
  exported by reviewers contain text and stay local; only ID + label files go
  into the repo (`labels/*.csv`, covered by `.gitignore`).
- The final labels are saved as `text_id,label` (a SHA-1 prefix of the
  normalized text), no text column.
- The split (Issue #2) is built from these labels with `RANDOM_SEED = 42`,
  stratified by class, same text kept in one split.

## Open items

1. **Inter-rater agreement** needs Gabriel's planned blind double-review
   (Issue #14).
2. **Consistency with meeting notes**: Issue #1 records "~1,000
   multi-specialty notes removed"; this pipeline keeps conflicting notes and
   re-labels them through review instead. The wording in the midterm Data
   section must match what was actually done.

## Where this dataset goes next

`labels/final_labels.csv` is the input for the rest of the modelling work:

1. **Frozen split (Issue #2):** the 2,230 notes are split into train /
   validation / test with `RANDOM_SEED = 42`, stratified by the 11 classes,
   with the same note text kept inside a single split. The text IDs already
   guarantee one row per unique text, so the split is built at the text level.
   The result is saved as `splits/split_seed42.csv` (text IDs + split column
   only, no text) and committed; every later notebook loads that file instead
   of re-splitting.

2. **Training:** the frozen split feeds the three model families: a
   statistical/trivial baseline, classical ML (TF-IDF + logistic regression
   and TF-IDF + linear SVM), and a fine-tuned ClinicalBERT. Evaluation uses
   macro-F1 as the primary metric with bootstrap confidence intervals; models
   are tuned on the validation split only, and the test split is touched once,
   at the end (Issues #2, #3, #22).

## Appendix: keyword tables and mappings

Every table below is defined and executed in `notebooks/02_eda.ipynb`
(Sections 4, 12.1, 12.4 and 12.5). They are repeated here so the cleaning can
be audited without running the notebook.

### A.1 Document-type labels dropped (Step 3)

Consult - History and Phy., SOAP / Chart / Progress Notes, Discharge Summary,
Emergency Room Reports, Office Notes, Letters.

### A.2 Raw label to final class mapping (Step 7)

| Raw label | Final class |
|---|---|
| Cardiovascular / Pulmonary | Cardiovascular / Pulmonary |
| Orthopedic, Podiatry | Orthopedic |
| Radiology | Radiology |
| General Medicine, Pain Management, Physical Medicine - Rehab, Sleep Medicine, Endocrinology, Rheumatology, Diets and Nutritions, Allergy / Immunology | General Medicine |
| Gastroenterology | Gastroenterology |
| Neurology, Neurosurgery | Neurology |
| Urology, Nephrology | Urology / Nephrology |
| Obstetrics / Gynecology, Pediatrics - Neonatal | Obstetrics / Gynecology / Pediatrics |
| ENT - Otolaryngology, Ophthalmology, Dentistry, Cosmetic / Plastic Surgery | ENT / Ophthalmology |
| Hematology - Oncology, Lab Medicine - Pathology | Hematology - Oncology |
| Psychiatry / Psychology | Psychiatry / Psychology |

Labels absent from this table (Autopsy, Bariatrics, Chiropractic, Dermatology,
Hospice - Palliative Care, IME-QME-Work Comp etc., Speech - Language) have no
specialty home and are excluded.

### A.3 General-surgery terms (Step 4, 21 notes)

debridement, incision and drainage, i&d, laceration, foreign body, eschar,
stab wound, dressing change, wound, soft tissue mass, chest wall mass,
central venous, g-tube, reintubat, fasciotom, pannus, hydradenitis, granuloma.

### A.4 Thyroid-surgery terms (Step 7, 7 notes)

thyroidectomy, hemithyroidectomy, lobectomy, thyroid mass, thyroid goiter,
goiter, central neck reoperation, paratracheal node, neck dissection, thyroid
surgery. (Applied only to texts that also carried a Surgery label.)

### A.5 Review suggestion tables (Step 6 first pass)

For each note the lowercased text is scanned table by table in this order and
the first specialty with a matching keyword wins; no match falls back to
General Medicine with low confidence. Confidence is rule-based: low = no
table matched, high = exactly one table matched, medium = two or more tables
matched.

| Specialty | Keywords |
|---|---|
| Radiology | x-ray, ct scan, mri, ultrasound, fluoroscop, angiogram, arteriogram, venogram, discogram, mammogram, sonogram, imaging, scan, c-arm |
| Orthopedic | fracture, arthroscop, bunion, knee, hip, shoulder, ankle, wrist, elbow, tendon, ligament, cartilage, osteotom, meniscus, rotator cuff, hallux, metatarsal, femur, tibia, fibula, humerus, clavicle, spondyl, fusion, laminectom, discectom, bone graft |
| Cardiovascular / Pulmonary | heart, cardiac, coronary, angioplasty, stent, aorta, aortic, vascular, artery, pulmonary, lung, thorac, cardiology, atrial, ventric, pacemaker, bypass, valve, echocardiogram |
| Gastroenterology | colon, colonoscop, endoscop, esophag, gastr, stomach, hepat, liver, bile, gallblad, cholang, pancreat, hernia, ercp, appendec, appendic, abdomen, rectum, anorectal, perirectal, hemorrhoid, diverticul, duoden, ileum, jejun, celiac, cholecyst, gastric |
| Neurology | brain, cranial, cervical spine, lumbar, nerve, seizure, epilep, headache, migraine, stroke, aneurysm, hydrocephalus, meningi, cerebro, spinal, disc, neuralg, neuropath, carpal, tremor, parkinson, craniotom, intracranial, dementia, alzheimer, lamin |
| Urology / Nephrology | kidney, renal, nephr, dialysis, bladder, prostate, prostat, urethr, ureter, stone, scrotal, testic, penile, circumcis, hydronephr, cystoscop, urologic |
| Obstetrics / Gynecology / Pediatrics | pregnan, obstetr, delivery, uterus, uterine, ovary, ovarian, cervix, fetal, neonat, pediatr, infant, newborn, placenta, endometri, fallopian, cesarean, amniot, gynecolog, c-section, prenatal |
| ENT / Ophthalmology | ear, hearing, audi, tympan, otitis, sinus, nasal, nose, throat, tonsil, laryng, pharyng, eye, ocular, cataract, retina, cornea, glaucoma, vision, vitrectom, strabismus, orbit, optic, otolaryng, septoplast, rhinoplast |
| Hematology - Oncology | cancer, carcinoma, tumor, tumour, leukem, lymphom, chemo, oncolog, malignan, metast, mastect, lumpectom, sarcoma, melanom, myelom, hodgkin, anemia, port-a-cath, port a cath, port insertion, biopsy |
| Psychiatry / Psychology | depression, depressive, anxiety, bipolar, schizo, suicid, psychiatr, psycholog, hallucin, delusion, adhd, mental health, mood disorder |

### A.6 Second-review corrections (Step 6)

Three first-review labels were flagged by the LLM assistant (two by reading the
28 unique-Surgery assignments, one by a keyword scan of the General Medicine
class) and corrected after confirmation by the modelling lead.

| text_id | Corrected label | Note |
|---|---|---|
| 99ea3545 | Gastroenterology | laparoscopic inguinal hernia repair (was General Medicine) |
| ac63df45 | DROP | buttock abscess incision-and-drainage (was General Medicine) |
| 38eddc9b | Obstetrics / Gynecology / Pediatrics | rectovaginal fistula closure (was Urology / Nephrology) |
