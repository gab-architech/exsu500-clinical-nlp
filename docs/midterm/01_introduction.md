> AI-drafted (Claude, planning thread, Nov 3). Owner: Jess to review and rewrite before Nov 4.
> Citations converted from [Author Year] to IEEE numbers (see `06_references.md`).
> FLAG: the research question below differs from the README's ("How much of the reported performance on this task comes from duplicates and label leakage, and how do model rankings change once these are controlled?"). Pick one and use it in both places.

# I. Introduction

Clinical documents must reach the right specialty team to be acted on, yet this routing step is a recognised point of failure. Physicians already spend 5.8 hours in the electronic health record for every 8 hours of scheduled patient care, of which 2.3 hours go to documentation [1]. When referrals and consultation notes are misdirected or delayed, care loops stay open: one primary care study identified 305 patients whose time-sensitive specialty referrals, including suspicious skin lesions and chest pain, were not completed on time [2], and breakdowns in information flow mean referring clinicians often receive specialist notes late or not at all [3]. Automating parts of this process is clinically desirable but hard: an AI triage system for ENT referrals agreed with historical categories only 53.8% of the time, largely because of inconsistent labels [4].

Natural language processing (NLP) offers a way to classify free-text notes by specialty. MTSamples, a public corpus of medical transcriptions, is widely used for this task, but recent work shows that earlier results were inflated by data leakage, and that on a leakage-free version the best model reaches a macro-F1 of only 0.279 [5].

We ask: how accurately can a clinical note's medical specialty be predicted from its free text alone, and how do model families of increasing complexity compare under a leakage-aware evaluation? Our contributions at this stage are (1) a cleaned, documented MTSamples dataset of 2,230 notes in 11 specialties; (2) manually validated labels with measured agreement (κ = 0.85); and (3) a pre-specified evaluation plan comparing five models across three families on a frozen, stratified split.
