# BRIDGE: Benchmarking Large Language Models for Understanding Real-world Clinical Practice Text

**Citation:** Wu, J., Gu, B., Zhou, R., et al. (2025). BRIDGE: Benchmarking large language models for understanding real-world clinical practice text. arXiv preprint arXiv:2504.19467. https://doi.org/10.48550/arXiv.2504.19467

**Read by:** Karman, Oct 29, 2026

**Peer-reviewed?** The 2025 version reviewed here is a preprint. A peer-reviewed version was published in *Nature Biomedical Engineering* in 2026.

**What they did:** Developed BRIDGE, a benchmark to evaluate LLMs on real clinical text rather than medical exam questions. They compared 95 LLMs across 87 tasks from 59 clinical datasets, covering nine languages and 14 medical specialties. Models were evaluated using zero-shot, five-shot, and chain-of-thought prompting.

**Main result:** The top overall benchmark score was 55.5 out of 100, showing that even leading models struggled with realistic clinical text. Five-shot prompting generally performed better than zero-shot and chain-of-thought prompting. Some open-source models performed similarly to proprietary models, while newer general-purpose LLMs often outperformed older medical-specific models.

**Limitation:** The benchmark used labels from the original datasets rather than having clinicians independently re-annotate them. Overall scores also combined different metrics, including accuracy, F1, and ROUGE-L, making comparisons across tasks more difficult. The study did not directly evaluate specialty classification using MTSamples.

**Relevance to our project:** This paper shows that strong performance on medical exams does not necessarily translate to real clinical text. It supports using careful data cleaning, checking specialty labels, and comparing models with the same evaluation methods. It also highlights how prompting strategies can affect LLM performance.
