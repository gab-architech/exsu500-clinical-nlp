# Text Categorization with Support Vector Machines

- **Citation:** T. Joachims, "Text Categorization with Support Vector Machines: Learning with Many Relevant Features," in Proc. ECML-98 (LNCS 1398), Chemnitz, Germany, 1998, pp. 137-142. DOI: 10.1007/BFb0026683.
- **Read by:** pending (draft written by the LLM assistant from the full text; fill in after a team member reads the paper)
- **Peer-reviewed?** yes

**What they did:** a theoretical and empirical analysis of why linear SVMs are well suited to text categorization, tested on the Reuters and Ohsumed corpora with TF-IDF features.

**Main result:** text classification has over 10,000 features per document, sparse document vectors, "dense" concepts (many words carry signal, so aggressive feature selection hurts), and most tasks are linearly separable - all of which favour a linear kernel. Linear SVMs outperformed the then-standard methods (naive Bayes, Rocchio, k-NN) across the corpora.

**Limitation:** results are from 1998 on news/abstract corpora, long before pretrained deep models existed; conclusions apply to bag-of-words linear settings.

**Relevance to our project:** the foundational justification for including TF-IDF + linear SVM in the classical ML family of our model comparison.
