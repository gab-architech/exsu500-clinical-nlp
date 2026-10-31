# Artificial Intelligence in Medical Referrals Triage Based on Clinical Prioritization Criteria

- **Citation:** Abdel-Hafez, A., Jones, M., Ebrahimabadi, M., Ryan, C., Graham, S., Slee, N., & Whitfield, B. (2023). Artificial intelligence in medical referrals triage based on Clinical Prioritization Criteria. *Frontiers in Digital Health*, 5, 1192975. https://doi.org/10.3389/fdgth.2023.1192975
- **Read by:** Karman, Oct 29, 2026
- **Peer-reviewed?** Yes, published in *Frontiers in Digital Health* (2023).

**What they did:** The authors evaluated different NLP and machine learning methods to classify 17,378 ENT referrals from two Queensland health services into three urgency categories. They compared text similarity methods, including Levenshtein distance, with KNN, supervised models (logistic regression, SVM, random forest, XGBoost), semi-supervised approaches, and K-means clustering.

**Main result:** Keyword-based Levenshtein similarity achieved the highest agreement with historical referral categories at 53.8%, compared to a random baseline of 33.3%. Most supervised models achieved agreement between 33% and 40%.

**Limitation:** Only 1,689 of the 17,378 referrals (9.7%) had historical CPC classifications, limiting the amount of labelled data available. Historical categories were also inconsistent with clinical guidelines in some cases, meaning agreement with previous classifications does not necessarily reflect clinical accuracy.

**Relevance to our project:** This paper shows the challenges of using NLP to classify clinical referrals when labels are limited or inconsistent. Although our project focuses on predicting medical specialties rather than urgency, it supports comparing simpler text-based approaches with more complex models and highlights the importance of validating our labels before training.
