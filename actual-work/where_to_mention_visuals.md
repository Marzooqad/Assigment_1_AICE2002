# Where to Mention Visuals in Your Report

## Visuals You Have:
- `confusion_matrices.png` / `confusion_matrices_smote.png` / `confusion_matrices_final.png`
- `step2_data_splitting.png`
- `step3_preprocessing.png`
- `hyperparameter_tuning_results.png`
- `performance_comparison.png`

## Where to Mention Them in Your Report:

### 1. **Section II (Quantifying Bias)**
After discussing class imbalance, add:
> "Figure X shows the class distribution across train/valid/test splits, illustrating the severe imbalance in both target variables."

### 2. **Section III (Task 1: Data Splitting)**
After explaining the speaker-based split, add:
> "Figure X visualizes the speaker distribution across splits, confirming no overlap between train/valid/test sets."

### 3. **Section IV (Task 2: Preprocessing)**
After discussing feature scaling, add:
> "Figure X shows the distribution of features before and after StandardScaler normalization, confirming zero mean and unit variance."

### 4. **Section VII (Task 5: Results)**
After Table I, add:
> "Figure X presents confusion matrices for all model-target combinations, revealing the per-class prediction patterns. The matrices clearly show the difficulty in detecting minority classes (lamb and wolf), with most models predicting the majority class (sheep) for target_asv."

Also add:
> "Figure X compares performance across all iterations, showing how F1-macro improved with each preprocessing step (baseline → SMOTE → aggressive weights → XGBoost)."

### 5. **Section VIII (Task 6: Hyperparameter Tuning)**
After discussing best parameters, add:
> "Figure X visualizes the hyperparameter search results, showing the performance landscape across different parameter combinations. The plot reveals that simpler models (fewer trees, lower C) performed better for this imbalanced dataset."

## Quick Format for Report:

**In your LaTeX/PDF, add figure captions like:**

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{confusion_matrices_final.png}
\caption{Confusion matrices for XGBoost and SVM on both target variables, showing per-class prediction patterns. The diagonal elements represent correct predictions, while off-diagonal elements show misclassifications.}
\label{fig:confusion}
\end{figure}
```

## Specific Recommendations:

1. **Confusion Matrices** → Section VII (Task 5) - Most important!
2. **Data Splitting Visualization** → Section III (Task 1)
3. **Preprocessing Visualization** → Section IV (Task 2)
4. **Hyperparameter Tuning Plot** → Section VIII (Task 6)
5. **Performance Comparison** → Section VII (Task 5) or Section IX (Answering Research Question)

## Don't Forget:
- Reference figures in text: "As shown in Figure X..."
- Number them sequentially: Figure 1, Figure 2, etc.
- Include captions explaining what each figure shows
- Make sure all figures are actually in your PDF when you submit!

