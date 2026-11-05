# Bag of Tricks for Efficient Text Classification

- **Citation:** A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov, "Bag of Tricks for Efficient Text Classification," in Proc. 15th Conf. European Chapter of the ACL (EACL 2017), Valencia, Spain, 2017, pp. 427-431. DOI: 10.18653/v1/E17-2068.
- **Read by:** Yuhan Zhang, Nov 5, 2026
- **Peer-reviewed?** yes

**What they did:** introduced fastText, a rank-constrained linear classifier over a bag of words plus hashed n-grams, with a hierarchical softmax for very large label spaces.

**Main result:** a linear model matched or beat deep models on several benchmarks (AG news 92.5 vs VDCNN 91.3; tied on Sogou and Yelp polarity), trained on over one billion words in under ten minutes on CPU, and was up to about 15,000x faster than deep classifiers. The paper argues that linear classifiers with the right features are strong text baselines and that deep networks' superiority on simple text tasks is not clearly established.

**Limitation:** ignores word order beyond local n-grams; trails VDCNN on a few sentiment datasets; does not use pretrained word embeddings.

**Relevance to our project:** the core evidence that TF-IDF + logistic regression is a strong baseline rather than a strawman, justifying the classical ML family in our comparison.
