# Evaluating LLM Workflows for Triage, Referral and Diagnosis

- **Citation:** F. Gaber et al., "Evaluating large language model workflows in clinical decision support for triage and referral and diagnosis," npj Digit. Med., vol. 8, Art. no. 263, 2025. DOI: 10.1038/s41746-025-01684-1.
- **Read by:** Yuhan Zhang, Nov 5, 2026
- **Peer-reviewed?** yes (Nature portfolio journal)

**What they did:** benchmarked LLM workflows (multiple model versions, including retrieval-augmented generation) on 2,000 MIMIC-derived clinical cases, evaluating triage level prediction, specialist referral (specialty) prediction and diagnosis suggestion.

**Main result:** the LLMs produced plausible referrals and diagnoses but over- and under-triaged, struggled with the most severe triage cases and with numeric laboratory values, and intra-model agreement varied; adding clinical context helped the stronger models but not the simpler ones.

**Limitation:** evaluates prompted/RAG LLMs only, with no fine-tuned supervised classifiers in the comparison; dataset is MIMIC-derived and curated.

**Relevance to our project:** the closest published task to ours (specialty referral from clinical notes). It supports our route decision: LLM prompting is not reliable enough for this task as-is, so we train task-specific supervised models instead, and this paper is the evidence that even large models mis-triage.
