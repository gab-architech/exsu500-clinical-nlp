
# Large Language Models to Identify Social Determinants of Health in Electronic Health Records

- **Citation:** Guevara, M., et al. (2024). Large language models to identify social determinants of health in electronic health records. *npj Digital Medicine*, 7, Article 6. https://doi.org/10.1038/s41746-023-00970-0
- **Read by:** Karman, Oct 29, 2026
- **Peer-reviewed?** Yes, published in *npj Digital Medicine* (2024).

**What they did:** The authors compared fine-tuned language models with zero- and few-shot GPT-3.5 and GPT-4 for extracting six social determinants of health (SDoH) from clinical notes. They used 800 radiotherapy notes for model development and evaluation, along with 200 immunotherapy and 200 MIMIC-III notes for additional testing. They also examined whether synthetic training data improved performance and how demographic information affected model predictions.

**Main result:** Fine-tuned Flan-T5 XXL achieved a macro-F1 of 0.71 for identifying any SDoH, while Flan-T5 XL achieved 0.70 for adverse SDoH. The fine-tuned models outperformed zero- and few-shot GPT-3.5 and GPT-4 across both tasks. They were also less sensitive to changes in demographic descriptors.

**Limitation:** The study focused on six SDoH categories in specific clinical populations, so the findings may not generalize to other classification tasks. There was also substantial class imbalance, with only around 3% of training sentences containing an SDoH mention, which made identifying less common categories more difficult.

**Relevance to our project:** This paper shows that fine-tuning models on clinical data can outperform larger LLMs without task-specific training. It supports comparing different model types using the same evaluation methods and highlights the importance of addressing class imbalance. Although the task differs from specialty classification, these findings are relevant to our model comparisons.
