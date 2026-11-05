> AI-drafted (Claude, planning thread, Nov 3; risks table revised with Claude Code, Nov 5). Owner: Gabriel.
> FLAG: fill in the 512-token share and the near-duplicate result once measured.

# V. Work Plan and Risks

*Roles.* Gabriel Barrere (data lead, coordination), Yuhan Zhang (modelling lead), Jessica Oentoro (writing lead), Karman Kang (interface lead, writing support). All members contribute code and text through a shared GitHub repository.

*Timeline.* Baseline and classical models are trained first (Nov 10 to 20), followed by ClinicalBERT fine-tuning with multiple seeds (Nov 16 to 27). The test set is evaluated once, after all models are finalised (Nov 27). The manuscript, model card and repository documentation follow (Nov 28 to Dec 6), then the recorded presentation (Dec 7 to 10), for submission on Dec 11.

*Risks and mitigations.*

| Risk | Mitigation |
|---|---|
| Small classes (smallest: 7 test notes) give unstable per-class scores | Macro-F1 as primary metric; bootstrap 95% CIs; per-class results reported |
| General Medicine merges 8 raw labels and is heterogeneous (14 blind-audit disagreements) | Per-class error analysis; General Medicine reported separately |
| Notes over 512 tokens are truncated by ClinicalBERT (share to be measured) | Truncation fixed in advance; compared with TF-IDF models, which have no length limit |
| Limited free GPU compute | Small number of seeds; saved checkpoints |
| Rule-based suggestions could anchor reviewers (48.5% match with final labels) | Reviewers read each full note and could override; blind audit measured agreement (κ = 0.85) |
| Near-duplicate notes across piles could inflate scores | Similarity check before training (result to be added) |
| MTSamples are sample transcriptions, not real routing data | Stated as a limit on generalisability |
