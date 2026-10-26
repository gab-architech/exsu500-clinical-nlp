# LLM Zero-shot Inference vs Task-specific Supervised Classification of Pathology Reports

- **Citation:** M. Sushil et al., "A comparative study of large language model-based zero-shot inference and task-specific supervised classification of breast cancer pathology reports," J. Amer. Med. Inform. Assoc., vol. 31, no. 10, pp. 2315-2327, 2024. DOI: 10.1093/jamia/ocae146.
- **Read by:** pending (draft written by the LLM assistant from the full text; fill in after a team member reads the paper)
- **Peer-reviewed?** yes (Q1 journal)

**What they did:** manually labelled 769 breast cancer pathology reports across 12 categories and compared zero-shot LLMs (GPT-4, GPT-3.5, Starling, ClinicalCamel) against supervised models (random forests, LSTM-Att, UCSF-BERT).

**Main result:** GPT-4 zero-shot reached an average macro-F1 of 0.86 versus 0.75 for the best supervised model (LSTM-Att), with the largest advantage on highly imbalanced categories; the other LLMs performed much worse. Supervised models remained comparable when a large labelled dataset is available.

**Limitation:** single cancer type and 12 categories; GPT-4 results are not reproducible (closed API, version drift); error analyses show GPT-4 over-relies on context while the supervised model failed mainly on generalization.

**Relevance to our project:** supports the choice between two routes: GPT-4 zero-shot is strong but API-dependent and not reproducible, so our proposal takes the supervised fine-tuning route (reproducible, free Colab budget) and cites this paper for that trade-off rather than claiming LLMs are weak.
