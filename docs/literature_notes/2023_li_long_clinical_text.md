# A Comparative Study of Pretrained Language Models for Long Clinical Text

- **Citation:** Li, Y., Wehbe, R. M., Ahmad, F. S., Wang, H., & Luo, Y. (2023). A comparative study of pretrained language models for long clinical text. *Journal of the American Medical Informatics Association*, 30(2), 295–304. https://doi.org/10.1093/jamia/ocac225
- **Read by:** Karman, Nov 5, 2026
- **Peer-reviewed?** Yes, published in *JAMIA*.

**What they did:** The authors pretrained two models, Clinical-Longformer and Clinical-BigBird, using approximately 2 million clinical notes from MIMIC-III. These models can process up to 4,096 tokens compared to the 512-token limit of standard BERT models. They compared the models across 10 clinical NLP datasets, including MedNLI, i2b2 2006/2010/2012/2014, emrQA-Relation, emrQA-Medication, emrQA-HeartDisease, OpenI, and MIMIC-AKI. The tasks included clinical inference, information extraction, question answering, and classification.

**Main result:** Both Clinical-Longformer and Clinical-BigBird significantly outperformed shorter-sequence models, including ClinicalBERT and BioBERT, across all 10 tasks. Their ability to process longer documents helped preserve clinical information that could otherwise be lost when text is shortened to 512 tokens.

**Limitation:** The models require more computational resources, and the pretraining data came from a single institution through MIMIC-III. They also used a general-domain tokenizer rather than one specifically adapted to clinical language. The study did not evaluate medical specialty classification using MTSamples.

**Relevance to our project:** This study shows why document length is important when working with clinical notes. If notes exceed the input limit of our model, we could lose information relevant to predicting the correct specialty. It supports considering longer-sequence models while also accounting for the additional computational requirements. However, we would first need to examine the length distribution of our MTSamples notes to determine whether this is a significant issue.
