> AI-drafted (Claude Code, Nov 3). Owner: Gabriel Barrere to review and rewrite before Nov 4.
> Budget: about 0.7 page. Every number comes from `docs/midterm_numbers.md`.

# III. Data

**Source and licence.** We use MTSamples, 4,999 transcribed clinical notes with about 40 specialty labels, distributed on Kaggle under a CC0 public-domain licence [17]. Only the `transcription` field is used as model input: the `keywords` field contains the label in 97.9% of the rows that have keywords, so it is excluded.

**Cleaning.** Table 1 lists each step. We removed 33 rows with no text, 938 rows whose label is a document type rather than a specialty (e.g. Discharge Summary), and the 1,088 Surgery rows, 1,039 of which are copies of notes already filed under a specialty. Because 95.9% of raw rows duplicate another row's text, we collapsed the data to one example per unique normalized text (2,279 texts), so the same note can never appear in two piles. Labels were then mapped to 11 classes; 97 texts with no specialty home or no routing class were excluded, leaving 2,230 notes.

**Classes.** Figure 1 shows the final distribution. Related labels were merged (e.g. Neurology with Neurosurgery; General Medicine with Pain Management, Physical Medicine - Rehab and five other small labels). The largest class, Cardiovascular / Pulmonary, has 358 notes and the smallest, Psychiatry / Psychology, 50: a 7.2x imbalance.

**Manual validation.** 606 texts needed a human decision: 578 filed under two or more specialties and 28 surgery-only notes. Two reviewers each labelled 303 of them, one note at a time with the full text; 602 were assigned a class and 4 dropped. A keyword rule (not a trained model) proposed a starting label and agreed with the final human label in only 48.5% of cases. A second-review audit corrected 3 labels.

**Blind audit.** One team member, blind to the existing labels, relabelled a stratified sample of 220 notes. On the 60 human-reviewed notes, Cohen's kappa was 0.847 (95% bootstrap CI 0.742 to 0.941). On the 160 auto-mapped notes, the disagreement rate was 13.8% (95% Wilson CI 9.3% to 19.9%). Fourteen of these 22 disagreements are notes that our class definition deliberately merges into General Medicine (such as pain management and rehabilitation notes) and the blind reviewer placed under Orthopedic or Cardiovascular / Pulmonary; excluding them, the rate is 5.0% (2.6% to 9.6%). We kept the merge and report it as a definitional difference. No labels were changed.

**Split.** Before any modelling, the 2,230 notes were split 70/15/15 into training (1,560), validation (335) and test (335) sets, stratified by class with a fixed seed (42). The split is saved as note IDs and committed, and the test set is used once, for the final evaluation. The smallest class has 8 validation and 7 test notes, so its per-class results will carry wide confidence intervals.

**Limitations.** MTSamples contains sample transcriptions written for illustration, not documents from a real routing workflow, and its specialty labels reflect the site's filing rather than a clinical decision. The class definitions involve judgment calls, visible in the General Medicine and Radiology disagreements of the blind audit. The label string still appears inside 8.7% of transcriptions; we will measure its effect with a masked-text variant rather than remove it.
