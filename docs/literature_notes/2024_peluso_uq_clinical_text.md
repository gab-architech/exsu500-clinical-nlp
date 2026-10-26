# Deep Learning Uncertainty Quantification for Clinical Text Classification

- **Citation:** A. Peluso et al., "Deep learning uncertainty quantification for clinical text classification," J. Biomed. Inform., vol. 149, Art. no. 104576, 2024. DOI: 10.1016/j.jbi.2023.104576.
- **Read by:** pending (draft written by the LLM assistant from the full text; fill in after a team member reads the paper)
- **Peer-reviewed?** yes (Q1 journal)

**What they did:** applied post-hoc calibration and selective classification (abstention) methods to CNN and HiSAN models that classify cancer pathology reports from NCI SEER, benchmarking against an in-house deep abstaining classifier (DAC).

**Main result:** the proposed methods reached the target accuracy with a lower rejection (abstention) rate than the DAC on both validation and holdout test data, required no retraining, and reduced the computational cost; calibrated confidence enables trusting high-confidence predictions while sending the uncertain ones to human review.

**Limitation:** evaluated on SEER pathology reports only; out-of-distribution behaviour was more complex; the DAC's earlier gains came mostly from majority classes with high rejection on minority classes.

**Relevance to our project:** supports our evaluation protocol (macro-F1 with bootstrap confidence intervals plus a reliability/calibration artifact) - reporting uncertainty for clinical text classifiers is a standard requirement, not an optional extra, and this paper is the citation for it.
