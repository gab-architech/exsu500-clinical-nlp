> AI-drafted (Claude, planning thread, Nov 3). Owner: Jess to finalise; each member must confirm their own CRediT line.
> Resolved Nov 5: Yuhan confirmed his CRediT line and his GenAI entry; the labelling tool used a keyword rule, not a model, so that false bullet was removed.

# CRediT author statement

- **Gabriel Barrere:** Conceptualization, Data curation, Formal analysis, Project administration, Software, Validation, Visualization, Writing (original draft), Writing (review and editing).
- **Yuhan Zhang:** Conceptualization, Data curation, Formal analysis, Methodology, Software, Validation, Writing (original draft).
- **Jessica Oentoro:** Conceptualization, Investigation, Validation, Writing (original draft), Writing (review and editing).
- **Karman Kang:** Investigation, Validation, Writing (original draft), Writing (review and editing).

# Use of Generative AI statement

**Claude (Anthropic, Claude Opus 5.5), via the Claude desktop app and Claude Code:** assisted with an initial literature search and draft summaries of candidate papers across the project, and produced reference first drafts of the report sections, which section owners compared with their own drafts and rewrote. It also assisted with the implementation and debugging of code for blind-review statistics, data splitting, near-duplicate detection and Figure 1, and supported the preparation of the project's numbers sheet and Table I, document assembly using the IEEE template, formatting, length adjustments, citation renumbering, repository administration and issue tracking.

**Claude Code (Anthropic CLI), model backend selected per session (claude-sonnet-5.5 and kimi-k3), Oct 15 to Nov 5:** assisted with coding and debugging the data-cleaning pipeline and the two offline review tools, drafted literature notes and the model proposal, which the modelling lead reviewed and finalized, and flagged three suspicious first-pass labels in a targeted audit. Session logs are available to the instructors upon request.

No generative model assigned a final label. The starting label suggestions shown to reviewers came from a deterministic keyword rule, which agreed with the final human label in 48.5% of cases. The 606 manual label decisions, the blind audit of 220 notes and the decision on each of the three flagged labels were made by team members.

*Verification:* every cited paper was read by a named team member, who checked its summary against the full text. Every section was reviewed and rewritten by its owner. All numbers were checked against the project's numbers sheet, generated from the repository code. Scripts were reviewed and their outputs inspected.
