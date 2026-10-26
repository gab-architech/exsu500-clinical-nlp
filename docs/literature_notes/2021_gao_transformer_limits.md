# Limitations of Transformers on Clinical Text Classification

- **Citation:** S. Gao et al., "Limitations of Transformers on Clinical Text Classification," IEEE J. Biomed. Health Inform., vol. 25, no. 9, pp. 3596-3607, 2021. DOI: 10.1109/JBHI.2021.3062322.
- **Read by:** pending (draft written by the LLM assistant from the full text; fill in after a team member reads the paper)
- **Peer-reviewed?** yes (Q1 journal)

**What they did:** compared BERT variants (truncation, segment pooling, target/multilabel attention) against simple word-level CNNs and a hierarchical self-attention network (HiSAN) on MIMIC-III discharge summaries (multilabel ICD coding, 42k documents) and SEER cancer pathology reports (six single-label tasks).

**Main result:** with full documents, no BERT variant significantly outperformed the simple models on any task; HiSAN had the best recall and F1, the CNN the best precision, and BERT was about 10x slower. The authors attribute this to attention being diluted away from the few keywords that determine clinical labels, and to WordPiece tokenization splitting the key terms that matter.

**Limitation:** only BlueBERT was tested; long-document BERT variants were capped at 10 segments; predates long-context domain transformers.

**Relevance to our project:** the strongest support for our research question - on clinical text a transformer does not automatically beat simple models, so our controlled comparison (rule baseline, TF-IDF linear models, ClinicalBERT) directly tests this on MTSamples.
