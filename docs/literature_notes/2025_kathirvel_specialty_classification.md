# Medical specialty classification with LLMs vs classical ML

- **Citation:** S. Kathirvel and L. Mookiah, "Medical Specialty Classification Using Large Language Models (LLMs)," Proc. Int. FLAIRS Conf., vol. 38, 2025. doi:10.32473/flairs.38.1.138953
- **Read by:** Gabriel Barrere, 2026-11-03
- **Peer-reviewed?** yes (conference proceedings)

**What they did:** compared BERT, BioBERT and DistilBERT with classical classifiers (including Naive Bayes) for predicting the specialty of medical transcription reports (MTSamples, about 5,000 records, 40 specialty categories).

**Main result:** Naive Bayes was best, with 86.16% accuracy and 84.52% F1, ahead of all transformer models (BERT about 63% accuracy).

**Limitation:** split procedure and duplicate handling are not reported, and accuracy is dominated by the large classes, so the numbers cannot be compared with leakage-free macro-F1 results.

**Relevance to our project:** an example of the high MTSamples scores that Barman 2026 attributes to leakage, and a reason to report macro-F1 with uncertainty and to document deduplication.
