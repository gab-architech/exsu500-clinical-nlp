# CLiGNet: Specialty Classification from Clinical Transcriptions (MTSamples)

- **Citation:** P. K. Barman and P. K. Barman, "CLiGNet: Clinical Label-Interaction Graph Network for Medical Specialty Classification from Clinical Transcriptions," arXiv:2603.22752, 2026. Preprint.
- **Read by:** Gabriel Barrere, 2026-11-03
- **Peer-reviewed?** no (preprint, not peer-reviewed)

**What they did:** built a leakage-free benchmark on MTSamples itself (4,966 records, 40 specialties, imbalance 181:1), using a Bio_ClinicalBERT encoder plus a graph network over a specialty label graph, with focal loss and inverse-frequency weighting instead of SMOTE. They explicitly critique earlier MTSamples papers that applied SMOTE before the train/test split.

**Main result:** the best leakage-free macro-F1 across seven baselines is only 0.279 (previous papers reported accuracy above 0.90); the label graph is the largest single gain (+0.066 macro-F1); calibration ECE 0.007. The paper includes McNemar testing and a failure analysis.

**Limitation:** preprint, not peer-reviewed; their 40-class single-label setup differs from our 11-class setup, so the absolute numbers are not directly comparable to ours.

**Relevance to our project:** direct support for our research question - reported MTSamples performance is largely leakage - and for our leakage handling (deduplicate to unique texts and keep the same text inside one split). Also independently confirms Bio_ClinicalBERT as the encoder of choice for this dataset.
