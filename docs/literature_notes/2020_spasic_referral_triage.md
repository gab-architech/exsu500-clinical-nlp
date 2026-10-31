# Patient Triage by Topic Modeling of Referral Letters: Feasibility Study

- **Citation:** Spasic, I., & Button, K. (2020). Patient triage by topic modeling of referral letters: Feasibility study. *JMIR Medical Informatics*, 8(11), e21252. https://doi.org/10.2196/21252
- **Read by:** Karman, Oct 29, 2026
- **Peer-reviewed?** Yes, published in *JMIR Medical Informatics* (2020).

**What they did:** The authors used Latent Dirichlet Allocation (LDA) topic modeling to identify patterns in referral letters for patients with knee or hip pain. The full cohort included 891 patients/referral letters with 1,048 recorded treatment outcomes. For the analysis, they used 576 patients with complete follow-up data and 944 treatment outcomes, training separate binary classifiers to predict different treatment pathways.

**Main result:** The topic-based classifiers generally performed better than a stratified random baseline at predicting treatment outcomes. Adding domain-specific clinical concepts improved performance, and the topics were also considered clinically interpretable.

**Limitation:** The study focused only on knee and hip referrals, limiting its generalizability to other specialties. Some treatment outcomes were uncommon, and high classification accuracy for these categories could reflect class imbalance rather than reliable prediction. The model was also not evaluated prospectively in clinical practice.

**Relevance to our project:** This paper shows how NLP can extract useful information from referral letters to support clinical triage. Although our project predicts medical specialties rather than treatment outcomes, it supports the use of simpler text-based approaches and shows why class imbalance and careful evaluation are important.
