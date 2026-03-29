PLAIN ENGLISH:

“The first to state his case seems right, until another comes and cross-examines him.” — Proverbs 18:17

Most people think predicting a lawsuit’s outcome means predicting who wins in court. That is the wrong question — and the numbers show exactly why. 
Of the 119,877 federal consumer banking cases in this dataset, only 26% ever received a court ruling at all. The remaining 74% settled (35.5%), were voluntarily dismissed (22.7%), 
or closed through other procedural mechanisms (15.7%). And of the cases that did reach adjudication, only 6.1% of the total — roughly 23% of adjudicated cases — resulted in a 
court-entered judgment for the plaintiff. A model that only looks at adjudicated outcomes is missing three-quarters of what actually happens. The banks told their story first. For decades, 
the 6.1% stood unexamined, unmeasured, accepted as the natural order of things. This model is the cross-examination. It applies just weights to a false balance that has disadvantaged consumers 
in federal court for a generation. Version 5.0 solves this in three ways. First, it separates structural population probability from case-specific probability. The neural network tells you 
what happens to the average case with 'your' profile across 119,877 real federal court records spanning all twelve circuits from 2011 through 2025. The weighted doctrinal engine tells you what 
happens to your specific case given documented concrete injury, prior regulatory action, defendant litigation history, and which circuit you are in. Second, it corrects for selection bias 
through a Heckman Two-Step correction. Cases that go to trial are not a random sample — they are the cases that could not settle, usually because the facts were too extreme or the 
defendant refused to pay. A logistic regression predicts P(settlement) from structural features including case duration, procedural stage, class action status, arbitration, and MDL consolidation;
the inverse of that probability becomes a sample weight for the adjudication head, upweighting the hard cases and test cases that generate disproportionate precedent and correcting the 
distortion inherent in training only on adjudicated outcomes. Third, it answers the question practitioners actually ask: not “will we win?” but “what is this case worth?” 
A case can carry low adjudicated win probability and still command a high settlement because of discovery exposure, class certification risk, or regulatory attention. The model produces a 
full five-outcome distribution across every case — high-value settlement, low-value settlement, plaintiff judgment, defendant judgment, dismissal — and flags cases as HIGH EXPOSURE when 
structural settlement pressure is high and adjudicated probability is non-trivial.

You're welcome.


TECHNICAL:

“A false balance is an abomination to the LORD, but a just weight is his delight.” — Proverbs 11:1

Technically, the model is a TensorFlow/Keras multi-head neural network trained on 119,877 FJC IDB consumer banking cases (NOS 480/190/371/370, 2011–2025, all 12 circuits). The shared trunk processes a 32-dimensional 
input vector encoding eight FJC predictor variables through three fully connected hidden layers — Dense(128) → BatchNorm → ReLU → Dropout(0.30); Dense(64) → BatchNorm → ReLU → Dropout(0.25); Dense(32) → 
BatchNorm → ReLU → Dropout(0.20) — feeding three specialized output heads: Head 1 (plaintiff_favorable) produces P(adjudicated plaintiff win) via custom weighted binary crossentropy loss (pos_weight=4.0, neg_weight=0.532), 
replacing class_weight which is unsupported for multi-output models in Keras/TF 2.19; Head 2 (settlement_pressure) is MSE-trained on a duration × PROCSTAT composite score; Head 3 (risk_score) concatenates the trunk 
with both head outputs to produce a unified risk/value signal. The weighted doctrinal engine is a Python port of server/prediction-engine.ts, calibrated to actual FJC JUDGMENT==1 plaintiff-favorable rates across 
all circuits and NOS codes, with a hostile-circuit scoring baseline of 0.366 and NOS-specific settlement range formulas incorporating concrete injury (2.5×), prior regulatory action (1.5×), and defendant institutional 
multipliers. The blended outcome model combines the neural network’s adjudicated probability with population base rates to produce the five-outcome distribution. Training converges at epoch 42 (best weights from epoch 27)
with early stopping monitoring val_plaintiff_favorable_auc. Validated AUC of 0.781 on a 23,976-case holdout. Distribution stability confirmed on a stratified 20,000-case out-of-sample dataset — 
all NOS deltas within ±0.3 percentage points actual, all circuit deltas within ±2 percentage points. Target variable: JUDGMENT==1 (6.1% positive rate). 

And no synthetic data.






