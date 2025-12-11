# Report Updates Guide
## How to Add Noise Experiment, Visuals, and Update Conclusion

---

## 1. WHERE TO ADD NOISE EXPERIMENT

### Option A: Add as a Subsection in Task 2 (Recommended)

**Location:** Section IV (Task 2: Preprocessing) - Add after "Step 4: Switching to XGBoost"

**New Subsection Title:**
```
E. Step 5: Adding Gaussian Noise (Data Augmentation Experiment)
```

**Content to Add:**
```
To test model robustness and explore data augmentation techniques, 
Gaussian noise (σ=0.05) was added to the training data after SMOTE 
resampling (implemented in step4 final xgboost with noise.py, Lines 
116-120). This approach adds noise to the already-balanced SMOTE 
samples, simulating real-world measurement variability without 
disrupting the class balance achieved by SMOTE.

Results: The noise experiment showed minimal impact on overall 
performance. XGBoost achieved F1-macro of 0.289 on target_asv 
(identical to the no-noise version), while SVM showed slight 
improvement (F1-macro: 0.252 vs 0.234). Wolf detection remained 
similar (28-32% recall), and lamb detection remained at 0%. This 
suggests that the classification challenge is not due to overfitting 
to clean data, but rather fundamental limitations in feature 
discriminability, particularly for distinguishing lamb from sheep.
```

**Line Reference:** Add to your references section:
```
[10] step4 final xgboost with noise.py. Gaussian noise augmentation 
experiment with noise added after SMOTE resampling (Lines 116-120).
```

---

### Option B: Add in Task 7 / Conclusion Section

**Location:** Section X (Limitations) or Section XI (Conclusion)

**Add this paragraph:**
```
Additional Experiment - Noise Augmentation: As a supplementary 
experiment, Gaussian noise (σ=0.05) was added to training data 
after SMOTE resampling to test model robustness (step4 final xgboost 
with noise.py). Results showed minimal performance change (F1-macro: 
0.289 vs 0.289 for XGBoost on target_asv), confirming that the 
classification difficulties stem from feature limitations rather than 
overfitting. This experiment demonstrates that even data augmentation 
techniques cannot overcome the fundamental challenge of distinguishing 
lamb from sheep using the provided 88 audio features.
```

---

## 2. WHERE TO MENTION VISUALS

### Visual 1: Confusion Matrices
**Location:** Section VII (Task 5: Results) - After Table I

**Add:**
```
Figure 1 presents confusion matrices for all model-target combinations, 
revealing the per-class prediction patterns. The matrices clearly show 
the difficulty in detecting minority classes (lamb and wolf), with most 
models predicting the majority class (sheep) for target_asv. Notably, 
the diagonal elements (correct predictions) are strongest for sheep 
and goat classes, while lamb and wolf show zero or minimal detection 
rates, confirming the severe class imbalance challenge.
```

**In LaTeX:**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.9\textwidth]{confusion_matrices_final.png}
\caption{Confusion matrices for XGBoost and SVM on both target variables, 
showing per-class prediction patterns. Diagonal elements represent correct 
predictions, while off-diagonal elements show misclassifications.}
\label{fig:confusion}
\end{figure}
```

---

### Visual 2: Data Splitting Visualization
**Location:** Section III (Task 1) - After explaining the split

**Add:**
```
Figure 2 visualizes the speaker distribution across train/valid/test 
splits, confirming no overlap between sets. The visualization also 
shows the class distribution within each split, highlighting the 
imbalance that persists across all data partitions.
```

**In LaTeX:**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{step2_data_splitting.png}
\caption{Speaker-based data splitting visualization showing distribution 
across train/valid/test sets and class distributions within each split.}
\label{fig:splitting}
\end{figure}
```

---

### Visual 3: Preprocessing Visualization
**Location:** Section IV (Task 2) - After discussing StandardScaler

**Add:**
```
Figure 3 shows the distribution of features before and after 
StandardScaler normalization, confirming that scaled features have 
approximately zero mean and unit variance, as required for SVM and 
beneficial for XGBoost.
```

**In LaTeX:**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{step3_preprocessing.png}
\caption{Feature distributions before and after StandardScaler 
normalization, showing the transformation to zero mean and unit variance.}
\label{fig:preprocessing}
\end{figure}
```

---

### Visual 4: Hyperparameter Tuning Results
**Location:** Section VIII (Task 6) - After discussing best parameters

**Add:**
```
Figure 4 visualizes the hyperparameter search results, showing the 
performance landscape across different parameter combinations. The 
plot reveals that simpler models (fewer trees, lower C values) 
performed better for this imbalanced dataset, suggesting that 
regularization helps prevent overfitting to the majority class.
```

**In LaTeX:**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{hyperparameter_tuning_results.png}
\caption{Hyperparameter tuning results showing F1-macro scores across 
different parameter combinations for Random Forest and SVM.}
\label{fig:hyperparams}
\end{figure}
```

---

### Visual 5: Performance Comparison
**Location:** Section VII (Task 5) or Section IX (Answering Research Question)

**Add:**
```
Figure 5 compares performance across all iterations, showing how F1-macro 
improved with each preprocessing step (baseline → SMOTE → aggressive 
weights → XGBoost). The progression demonstrates the iterative refinement 
process and shows that while improvements were made, the fundamental 
challenge of class imbalance remains.
```

**In LaTeX:**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{performance_comparison.png}
\caption{Performance evolution across all experimental iterations, 
comparing F1-macro scores for different approaches.}
\label{fig:performance}
\end{figure}
```

---

## 3. WHAT TO CHANGE IN CONCLUSION

### Current Conclusion Issues:
- Doesn't mention the noise experiment
- Could emphasize the iterative process more
- Could be clearer about what worked vs what didn't

### Updated Conclusion (Section XI):

**Replace your current conclusion with:**

```
Both target_human and target_asv present classification challenges 
under severe class imbalance, with F1-macro scores ranging from 0.19 
to 0.31 across the experiments. Target_human proved to be slightly 
more difficult (F1-macro: 0.19-0.24) compared to target_asv (0.23-0.29), 
but the difference is small enough that both labeling schemes prove 
similarly challenging.

The real story here is about iterative refinement and learning from 
failure. The journey from baseline class weighting through SMOTE to 
aggressive manual weights to XGBoost shows how trying different approaches 
when standard methods don't work is necessary. Successfully improved 
wolf class detection from 0% to 32.26% recall using XGBoost with 100x 
class weights, which demonstrates that the techniques do help, just not 
enough to overcome limitations in the features.

The persistent inability to detect lamb despite extreme measures (100x 
class weights, thousands of synthetic SMOTE samples, XGBoost's iterative 
error correction, and even noise augmentation experiments) suggests 
there was a fundamental limitation that these preprocessing and algorithmic 
techniques cannot break through. The features themselves probably don't 
encode the information needed to distinguish lamb from sheep.

This experience reinforces an important lesson: choosing the right 
evaluation metrics matters enormously. If the study only looked at 
accuracy, it would appear that the baseline models were doing okay at 
40-65% accuracy. F1-macro revealed the truth—they were mostly just 
guessing sheep.

The marginal difference between target_human and target_asv answers the 
research question, but perhaps the more valuable insight is understanding 
the limitations of the different approaches and what would be needed to 
push performance further. The noise augmentation experiment (Section IV.E) 
further confirmed that the challenge is not overfitting, but rather 
fundamental feature limitations.
```

---

## 4. QUICK CHECKLIST FOR REPORT UPDATES

### Task 2 (Preprocessing):
- [ ] Add subsection "E. Step 5: Adding Gaussian Noise"
- [ ] Mention line numbers: Lines 116-120 of step4 final xgboost with noise.py
- [ ] Add reference [10] in references section

### Task 5 (Results):
- [ ] Add Figure 1 (confusion matrices) after Table I
- [ ] Add Figure 5 (performance comparison) 
- [ ] Reference figures in text: "As shown in Figure 1..."

### Task 1 (Data Splitting):
- [ ] Add Figure 2 (data splitting visualization)
- [ ] Reference in text after explaining split

### Task 2 (Preprocessing):
- [ ] Add Figure 3 (preprocessing visualization)
- [ ] Reference after StandardScaler discussion

### Task 6 (Hyperparameter Tuning):
- [ ] Add Figure 4 (hyperparameter tuning results)
- [ ] Reference after discussing best parameters

### Conclusion:
- [ ] Update to mention noise experiment
- [ ] Emphasize iterative process
- [ ] Reference Section IV.E for noise experiment

### References:
- [ ] Add [10] for noise experiment code

---

## 5. EXAMPLE FIGURE PLACEMENT IN REPORT STRUCTURE

```
I. Introduction
II. Quantifying Bias in Our Data
III. Task 1: How the Data Was Split
    → Figure 2: Data Splitting Visualization
IV. Task 2: Preprocessing - A Journey of Iteration
    A. Step 1: The Baseline
    B. Step 2: Adding SMOTE
    C. Step 3: SMOTE Plus Aggressive Weights
    D. Step 4: Switching to XGBoost
    E. Step 5: Adding Gaussian Noise (NEW)
    → Figure 3: Preprocessing Visualization
V. Task 3: Why Were These Algorithms Chosen
VI. Task 4: Combining Systems
VII. Task 5: Results and What They Mean
    → Table I: Performance Evolution
    → Figure 1: Confusion Matrices
    → Figure 5: Performance Comparison
VIII. Task 6: Hyperparameter Tuning
    → Figure 4: Hyperparameter Tuning Results
IX. Answering the Research Question
X. Limitations and What Could Be Done Differently
XI. Conclusion (UPDATED)
References
    → Add [10] for noise experiment
```

---

## 6. SAMPLE TEXT FOR NOISE EXPERIMENT IN TASK 2

**Add this as Section IV.E:**

```
E. Step 5: Adding Gaussian Noise (Data Augmentation Experiment)

As a supplementary experiment to test model robustness, Gaussian noise 
(σ=0.05) was added to training data after SMOTE resampling (implemented 
in Lines 116-120 of step4 final xgboost with noise.py). This approach 
ensures SMOTE first creates balanced synthetic samples from clean data, 
then noise is added as a regularization technique to simulate real-world 
measurement variability.

The noise was applied only to the resampled training data, not to 
validation or test sets, to maintain proper evaluation. The standard 
deviation of 0.05 (5% of the feature scale) was chosen to be small 
enough to preserve feature information while introducing meaningful 
variation.

Results showed minimal impact on overall performance:
- XGBoost on target_asv: F1-macro remained at 0.289 (identical to 
  no-noise version)
- SVM on target_asv: F1-macro improved slightly to 0.252 (vs 0.234 
  without noise)
- Wolf detection: Maintained 28-32% recall, similar to no-noise version
- Lamb detection: Remained at 0% recall

This experiment confirms that the classification challenge is not due 
to overfitting to clean data, but rather fundamental limitations in 
the discriminability of the 88 audio features, particularly for 
distinguishing lamb from sheep. The noise augmentation did not provide 
the breakthrough needed to detect lamb, further supporting the 
conclusion that different features or approaches would be required.
```

---

*Use this guide to update your report systematically!*

