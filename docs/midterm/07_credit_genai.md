> AI-drafted (Claude, planning thread, Nov 3). Owner: Jess to finalise; each member must confirm their own CRediT line.
> Resolved Nov 5: Yuhan confirmed his CRediT line and his GenAI entry; the labelling tool used a keyword rule, not a model, so that false bullet was removed.

# CRediT author statement

- **Gabriel Barrere:** Conceptualization, Data curation, Formal analysis, Project administration, Software, Validation, Visualization, Writing (original draft), Writing (review and editing).
- **Yuhan Zhang:** Conceptualization, Data curation, Formal analysis, Methodology, Software, Validation, Writing (original draft).
- **Jessica Oentoro:** Conceptualization, Investigation, Validation, Writing (original draft), Writing (review and editing).
- **Karman Kang:** Investigation, Validation, Writing (original draft), Writing (review and editing).

# Use of Generative AI statement

The team used generative AI tools as follows:

- **Claude (Anthropic, Claude Opus 5.5), via the Claude desktop app and Claude Code:** helped with an initial literature search and drafted summaries of candidate papers; every cited paper was then read in full by a named team member, who verified or rewrote its note; reference first drafts of all report sections, which section owners compared and merged with their own drafts; code for the blind-review statistics, data split and Figure 1; the numbers sheet and Table 1; compilation of the review copy; repository administration and issue tracking.
- **Claude Code (Anthropic CLI), model backend selected per session (kimi-k3 and other models), Oct 15 to Nov 5:** data cleaning analysis and the cleaning pipeline, the two offline review tools, drafts of literature notes and the model proposal, each then reviewed and finalized by the modelling lead, and a targeted audit of the first-pass labels that flagged three suspicious labels. The modelling lead confirmed each correction. Session logs are available to the instructors on request.
- No generative model assigned a final label. The starting label suggestions shown to reviewers came from a deterministic keyword rule, which agreed with the final human label in 48.5% of cases.

Verification: every cited paper was read by a named team member, who corrected its summary against the full text. Every section was reviewed and rewritten by its owner. All numbers were checked against the project's numbers sheet, generated from the repository code. Scripts were reviewed and their outputs inspected. Label suggestions were reviewed by humans, the three flagged labels were checked against the note text and confirmed by the modelling lead, and a blind audit of 220 notes measured label agreement (Cohen's kappa 0.847 on the human-reviewed stratum). The authors take full responsibility for the content of this report.

**Not AI-assisted:** the 606 manual label decisions, the blind audit of 220 notes, and the three correction decisions were made by team members. AI produced drafts, analysis code and audit flags only, and no AI output was submitted without review by its owner.
