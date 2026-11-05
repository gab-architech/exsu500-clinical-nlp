# NLP for Information Extraction from EHRs in Cancer: Systematic Review

- **Citation:** S. Dahl, M. Bøgsted, T. Sagi, and C. Vesteghem, "Performance of Natural Language Processing for Information Extraction From Electronic Health Records Within Cancer: Systematic Review," JMIR Med. Inform., vol. 13, Art. no. e68707, 2025. DOI: 10.2196/68707.
- **Read by:** Yuhan Zhang, Nov 5, 2026
- **Peer-reviewed?** yes

**What they did:** systematic review of 33 studies extracting information from cancer EHR text, categorizing every model as rule-based, traditional ML, CRF, neural network or bidirectional transformer, and aggregating per-study best F1.

**Main result:** best-per-article F1 ranged from 0.355 to 0.985; transformers outperformed every other model family (mean F1 advantage 0.044 to 0.234), and transformer use increased over time.

**Limitation:** heterogeneous tasks, datasets and label definitions make the aggregated numbers rough; review is limited to cancer applications.

**Relevance to our project:** supports our study design - comparing model families explicitly is the standard approach in the field - and sets the prior expectation that transformers usually win, which makes our controlled comparison (and any exception, as in Gao 2021) informative rather than obvious.
