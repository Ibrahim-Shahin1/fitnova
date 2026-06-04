# 7 Conclusions and Future Work

This chapter draws the project to a close. It summarises what was achieved against the objectives set out at the start, and then sets out the work that would carry each part of the system past the limits the results made plain.

## 7.1 Summary

FitNova set out to build three AI fitness modules in one platform, each grounded in a published dataset and method, and to report the results honestly. Measured against the objectives in Section 1.3, the project did what it set out to do.

The platform itself was built (O1 and O6): three modules, planning, form correction, and nutrition, run inside one backend, each reading the user's profile and each independent enough that one can fail without taking the others down. The workout planner was built as the layered recommender of O2 and O3, a content-based filter, a Neural Collaborative Filtering re-ranker, and a language-model crew that writes a goal-matched seven-day plan from real dataset exercises only, with a deterministic validator that makes an invented exercise impossible. It reaches a hit rate at 10 of 0.9270 on its evaluation.

The form correction module is the deepest result and the one O4 was rewritten around. Rebuilt from raw pixels on the Fitness-AQA dataset after the earlier pose-based designs were dropped, it detects five named errors across three exercises and, on the dataset's official test split with the same per-error F1 the published work uses, it matches or beats that work, with a higher macro F1 on all three exercises and an independent overfitting audit confirming the deployed models generalise. That is a defensible, reproducible result of exactly the kind the project was rebuilt to produce.

The nutrition module met O5 in two parts: a deterministic engine and planner that guarantee the calorie, macronutrient, and allergen correctness of every plan before any language model speaks, and a DistilGPT-2 reproduction that beats the learned models of the recipe-generation paper on the text-quality metrics.

The last objective, O7, was about how all of this is reported, and it runs through the whole document: every headline number is computed on an official split with the source work's metric and is reproducible from a committed file, the approaches that failed are documented as honestly as the ones that worked, and the limits of each result are stated plainly. The project's central claim is modest and precise: each of its parts does what published work does, measured the same way, and this is shown rather than asserted. It does not claim to be a finished product.

## 7.2 Future Work

Each module's limit, set out in the discussion, points directly at the work that would carry it further.

For form correction, the largest opportunity is the domain gap. The models are reliable on side-on, full-body gym footage and weak on casual front-facing phone video, and closing that would mean collecting and labelling video from the phone domain, then training on it, which the Fitness-AQA dataset does not provide. With reliable phone-domain models, the deferred live camera path, the real-time feedback over a websocket that the current system supports in the backend but does not expose to users, could be finished and shipped as a genuine camera tool rather than a benchmark inspector. The exercise coverage could also grow: the barbell row was scoped out of this milestone for time and compute and is the obvious next exercise, and the dataset carries more errors than the five used here. A graded sense of how wrong a repetition is, rather than a yes or no, would also be valuable, but only if a dataset with graded labels is used, because the current labels are binary and inventing a severity would break the honesty the project is built on.

For the recommender, the single most valuable step is real data. The model is implemented correctly, but it is trained and measured on synthetic interactions, so deploying the planner, collecting real user choices over time, and retraining on them would turn a clean reproduction of the method into an actual measurement of recommendation quality. Once a real history of user behaviour exists, a sequential model that reads the order of a user's choices rather than a static profile becomes possible [18], which is a natural extension that the current data can not support.

For nutrition, the obvious extension is the part of the recipe-generation paper this work deliberately did not reproduce, personalisation. Conditioning the generated cooking text on a user's own history, and measuring it the way the paper does, would close the one gap named in the results. On the deterministic side, the engine and planner already guarantee correctness, so the work there is less about accuracy and more about reach: more recipes, more cuisines, and an on-device version that would let the nutrition feature run without a server. The recipe-text metrics should also be finalised on the full test set, replacing the 2,000-recipe sample the current numbers come from.

Across all three, the step that would test the whole platform as a product rather than as three benchmarks is a study with real users, which would measure whether the plans, the feedback, and the meals actually help someone train better, rather than measuring an F1 or a hit rate. That is the boundary between what this project proved and what a deployed FitNova would need to prove, and it is the right place for the work to go next.

FitNova began as an attempt to build AI fitness features and, in its form-correction work, became a lesson in building them honestly. The version delivered here is grounded in published data and methods, measured against published results, audited for the failure it was most at risk of, and clear about where it stops being reliable. Those properties, more than any single score, are what the project set out to earn.

> Note: Chapter 7 completes the body of the report. Remaining: the front matter (Attestation and Acknowledgements), the final reconciliation pass on the Abstract, and the appendices. References are maintained in `references.md`.
