# Publicly Available Clinical BERT Embeddings

- **Citation:** E. Alsentzer et al., "Publicly Available Clinical BERT Embeddings," in Proc. 2nd Clinical NLP Workshop (NAACL-HLT 2019), Minneapolis, MN, USA, 2019, pp. 72-78. DOI: 10.18653/v1/W19-1909.
- **Read by:** pending (draft written by the LLM assistant from the full text; fill in after a team member reads the paper)
- **Peer-reviewed?** yes (ACL Anthology workshop paper)

**What they did:** further-pretrained BERT on MIMIC-III clinical notes (about 2 million notes, 15 note types) and released four public models: Clinical BERT, Discharge Summary BERT, Bio+Clinical BERT and Bio+Discharge Summary BERT. Evaluated with a single linear classification head on MedNLI (clinical inference) and four i2b2 tasks (two de-identification, two concept/entity extraction).

**Main result:** Bio+Clinical BERT reached 82.7 accuracy on MedNLI vs 80.8 (BioBERT) and 77.6 (plain BERT); i2b2 2010 concept extraction F1 87.8 (+4.3 over BERT). However, the clinical models gave no gain on the two de-identification tasks (BioBERT stayed best, 94.8 F1), showing that domain pretraining is not automatically beneficial when pretraining and task text distributions differ.

**Limitation:** downstream tasks used only a single linear layer; MIMIC-III is one ICU at one institution; pretraining cost 17-18 GPU days.

**Relevance to our project:** supports our choice of `emilyalsentzer/Bio_ClinicalBERT` (the best variant on MedNLI), and supports the caveat that domain pretraining does not guarantee a win - which is exactly what our model comparison tests.
