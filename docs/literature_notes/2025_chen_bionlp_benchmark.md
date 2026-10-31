# Benchmarking Large Language Models for Biomedical Natural Language Processing Applications and Recommendations

- **Citation:** Chen, Q., et al. (2025). Benchmarking large language models for biomedical natural language processing applications and recommendations. Nature Communications, 16, Article 3280. https://doi.org/10.1038/s41467-025-56989-2
- **Read by:** Karman, Oct 29, 2026
- **Peer-reviewed?** - Yes, published in *Nature Communications* (2025).

**What they did:** Compared GPT-3.5, GPT-4, LLaMA 2 13B, and PMC-LLaMA 13B across 12 biomedical NLP datasets covering six applications: document classification, question answering, named entity recognition, relation extraction, text summarization, and text simplification. They evaluated zero-shot, few-shot, and fine-tuned approaches against conventional fine-tuned language models such as PubMedBERT, BioBERT, and BART.

**Main result:** Fine-tuned models generally performed better than zero- and few-shot LLMs, with a reported aggregate score of 0.6536 compared with 0.4561 for the best zero-shot approach, 0.4750 for one-shot, and 0.5131 for fine-tuned LLMs. Performance varied by task, with GPT-4 performing well on medical question answering.

**Limitation:** The study primarily evaluated biomedical NLP benchmarks rather than medical specialty classification on real clinical notes. Performance across different tasks was measured using different metrics, making aggregate scores difficult to interpret as a single classification measure.

**Relevance to our project:** This paper shows that larger LLMs do not necessarily perform better than smaller fine-tuned models. It supports comparing simpler approaches with more complex models using the same dataset and evaluation metrics, rather than assuming that model size will improve specialty classification.
