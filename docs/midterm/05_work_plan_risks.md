> AI-drafted (Claude, planning thread, Nov 3). Owner: Gabriel to review and rewrite before Nov 4.
> FLAG (row 5): label suggestions during review came from a keyword rule, not AI (`docs/data_cleaning.md` Step 6: "No LLM produced any label decision"). Rewrite as "Rule-based label suggestions could anchor reviewers".
> FLAG (row 3): "Many notes exceed 512 tokens" is not measured yet (`docs/model_proposal.md` says most notes fit). Soften or measure before Nov 9.

# V. Work Plan and Risks

*Roles.* Gabriel Barrere (data lead, coordination), Yuhan Zhang (modelling lead), Jessica Oentoro (writing lead), Karman Kang (interface lead, writing support). All members contribute code and text through a shared GitHub repository.

*Timeline.* Baseline and classical models are trained first (Nov 10 to 20), followed by ClinicalBERT fine-tuning with multiple seeds (Nov 16 to 27). The test set is evaluated once, after all models are finalised (Nov 27). The manuscript, model card and repository documentation follow (Nov 28 to Dec 6), then the recorded presentation (Dec 7 to 10), for submission on Dec 11.

*Risks and mitigations.*

| Risk | Mitigation |
|---|---|
| Small classes (smallest: 7 test notes) give unstable per-class scores | Macro-F1 as primary metric; bootstrap 95% CIs; per-class results reported |
| General Medicine merges 8 source labels and is heterogeneous (14 blind-audit disagreements) | Per-class error analysis; report General Medicine performance separately |
| Many notes exceed ClinicalBERT's 512-token limit | Truncation strategy fixed in advance; compared against TF-IDF models with no length limit |
| Limited free GPU compute | Small number of seeds; saved checkpoints |
| Label suggestions were AI-assisted during review | Blind human audit (κ = 0.85); disclosed in the GenAI statement |
| MTSamples are sample transcriptions, not real routing data | Stated as a limit on generalisability |
