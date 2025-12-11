# Assignment Requirements Check
## Comparing Assignment PDF Requirements vs Your Report

This document checks what the assignment requires against what's in your report.

---

## Assignment Requirements Summary

From **AICE2002 - Assignment 1.pdf**, the assignment requires:

### Core Question
- **Which labeling scheme (target_human or target_asv) is more challenging to predict?**

### Required Components
1. Compare classification performance for predicting target_human vs. target_asv
2. Use at least **two different algorithms**
3. Report results in a **table with at least two different performance metrics**
4. Include **additional tables, figures, and plots** to support decisions
5. For **Tasks 1-6**, indicate which lines of code correspond to each task

### Task Requirements

#### Task 1: Data Splitting
- How you divided dataset into train/valid/test splits and why
- How are label distributions represented in each split?
- Did you combine data from all three systems or keep them separate?
- Did you divide data by gender or speaker?
- What steps did you take to make the problem more tractable?
- Did you use k-fold cross-validation, and if so how many folds?

#### Task 2: Preprocessing
- How you pre-processed the dataset and why
- Did you normalise or standardise any features?
- Did you have missing values and if so, how did you handle these?
- Are the classes balanced or imbalanced and did you take steps to alter class balance?
- Did you add noise to the dataset, and if so, why and what kind?

#### Task 3: Algorithm Selection
- Which algorithms you chose and why
- Did you use a mix of supervised and unsupervised?
- Did you perform any feature selection or visualisation before selecting an algorithm?

#### Task 4: Handling Three Systems
- How you handled data across the three systems (orig, X, and Y)
- Are you reporting results for each system independently or combined as one large dataset?
- What are the benefits or tradeoffs from your decision?

#### Task 5: Evaluation Metrics
- Which evaluation metrics you chose to inform conclusions and why
- Which evaluation metrics were most useful for answering the research question?
- Is performance on one evaluation metric better than performance on another?
- Is performance on a particular category label better than another (e.g., easier to predict lamb and more challenging to predict wolf)?

#### Task 6: Hyperparameter Tuning
- What range of hyperparameters you tried for optimal performance
- If you tried different parameters, explain this, report values in a table, and indicate which ones performed best

#### Task 7: Summary and Conclusions
- A few sentences to summarise findings and conclusions
- Explain how your experiment design and results answer the original research question
- Is the assignment question answerable/solvable, why or why not?
- Does one algorithm perform better than another, why or why not?
- Are there other experiments you would run, or other ways you would test your design, if you had more time?

### Formatting Requirements
- **IEEE conference style** (template provided in Overleaf)
- Submit as **compiled PDF** (NOT LaTeX file)
- **Maximum 5-6 pages** (not including references)
- Clearly mark where each task is addressed (use subheadings or indicators)

---

## Your Report Coverage Analysis

### ✅ **FULLY COVERED**

#### Core Question
- ✅ **Addressed**: Section IX "ANSWERING THE RESEARCH QUESTION" clearly states target_human is slightly harder (0.19-0.24 vs 0.23-0.29 F1-macro)

#### Algorithms
- ✅ **Two algorithms used**: Random Forest and SVM (baseline), then XGBoost (final)
- ✅ **Justification provided**: Section V "TASK3: WHY WERE THESE ALGORITHMS CHOSEN"

#### Performance Metrics Table
- ✅ **Table I provided**: "HOW PERFORMANCE EVOLVED THROUGH THE ITERATIONS" with Accuracy and F1-macro
- ✅ **Additional metrics**: Per-class precision/recall, confusion matrices mentioned

#### Task 1: Data Splitting
- ✅ **Section III**: "TASK1: HOW THE DATA WAS SPLIT"
- ✅ **Speaker-based splitting**: Explained (Lines 61-100)
- ✅ **Combined systems**: Yes, all three systems combined (4,923 samples)
- ✅ **Split by speaker**: Yes, not by gender or random samples
- ✅ **Label distributions**: Mentioned in Section II (class imbalance ratios)
- ✅ **K-fold CV**: Addressed - explains why NOT used (Section III.C)
- ✅ **Line numbers**: Provided (Lines 61-100, 73-85, 131-180)

#### Task 2: Preprocessing
- ✅ **Section IV**: "TASK2: PREPROCESSING - A JOURNEY OF ITERATION"
- ✅ **Standardization**: StandardScaler applied (Lines 121-180)
- ✅ **Missing values**: Checked, none found (mentioned in preprocessing)
- ✅ **Class imbalance**: Severely imbalanced, addressed with SMOTE + class weights
- ✅ **Line numbers**: Provided (Lines 121-180, 110-120, 80-100)
- ⚠️ **Noise addition**: NOT mentioned (assignment asks if you added noise)

#### Task 3: Algorithm Selection
- ✅ **Section V**: "TASK3: WHY WERE THESE ALGORITHMS CHOSEN"
- ✅ **Algorithms justified**: Random Forest, SVM, XGBoost with reasons
- ✅ **Supervised only**: Yes, all supervised (no unsupervised mentioned)
- ⚠️ **Feature selection/visualization**: NOT explicitly mentioned before algorithm selection
- ✅ **Line numbers**: Provided (Lines 80-95, 96-105)

#### Task 4: Handling Three Systems
- ✅ **Section VI**: "TASK4: COMBINING SYSTEMS"
- ✅ **Decision explained**: Combined all three systems into one dataset
- ✅ **Benefits/tradeoffs**: Explained (3x more data vs potential bias)
- ✅ **Rationale**: Similar feature distributions across systems

#### Task 5: Evaluation Metrics
- ✅ **Section VII**: "TASK5: RESULTS AND WHAT THEY MEAN"
- ✅ **Metrics chosen**: F1-macro (primary), Accuracy, per-class precision/recall, confusion matrices
- ✅ **Why chosen**: Explained (F1-macro treats classes equally, crucial for imbalance)
- ✅ **Most useful**: F1-macro identified as most useful
- ✅ **Metric comparison**: Accuracy vs F1-macro discussed
- ✅ **Per-class performance**: Section VII.C discusses sheep, goat, wolf, lamb performance

#### Task 6: Hyperparameter Tuning
- ✅ **Section VIII**: "TASK6: HYPERPARAMETER TUNING"
- ✅ **GridSearchCV**: Used with 3-fold cross-validation
- ✅ **Parameter ranges**: Listed (RF: n_estimators, max_depth, min_samples_split, min_samples_leaf; SVM: C, gamma)
- ✅ **Best parameters**: Reported (RF: n_estimators=50, max_depth=10; SVM: C=0.1, gamma=0.001)
- ✅ **Line numbers**: Provided (step5 hyperparameter tuning.py)
- ⚠️ **Table of hyperparameters**: Values listed in text but not in a formal table format

#### Task 7: Summary and Conclusions
- ✅ **Section IX**: "ANSWERING THE RESEARCH QUESTION"
- ✅ **Section XI**: "CONCLUSION"
- ✅ **Findings summarized**: Target_human slightly harder, both difficult
- ✅ **Question answerability**: Addressed (yes, answerable but challenging)
- ✅ **Algorithm comparison**: XGBoost better than Random Forest for imbalance
- ✅ **Future experiments**: Section X "LIMITATIONS AND WHAT COULD BE DONE DIFFERENTLY"

#### Additional Requirements
- ✅ **Figures/plots**: Confusion matrices, performance comparisons mentioned
- ✅ **Tables**: Table I with performance metrics
- ✅ **Line numbers**: Provided for all tasks (though some need updating per mapping document)
- ✅ **IEEE style**: Appears to follow IEEE format
- ✅ **Page length**: Appears to be within 5-6 pages

---

## ⚠️ **POTENTIALLY MISSING OR WEAK AREAS**

### 1. **Task 2: Noise Addition**
- **Assignment asks**: "Did you add noise to the dataset, and if so, why and what kind?"
- **Your report**: Does NOT mention adding noise
- **Status**: If you didn't add noise, you should explicitly state this

### 2. **Task 3: Feature Selection/Visualization Before Algorithm Selection**
- **Assignment asks**: "Did you perform any feature selection or visualisation before selecting an algorithm?"
- **Your report**: Does NOT explicitly mention feature selection or visualization done BEFORE selecting algorithms
- **Status**: Should clarify if you did any exploratory analysis before choosing algorithms

### 3. **Task 6: Hyperparameter Table**
- **Assignment asks**: "report the values in a table, and indicate which ones performed best"
- **Your report**: Lists parameter ranges and best values in text, but not in a formal table
- **Status**: Consider adding a table with all tried hyperparameters and best values

### 4. **Task 5: Explicit Category Comparison**
- **Assignment asks**: "Is performance on a particular category label better than another (e.g., easier to predict lamb and more challenging to predict wolf)?"
- **Your report**: Section VII.C discusses this, but could be more explicit
- **Status**: Covered but could be clearer

### 5. **Visualizations/Figures**
- **Assignment requires**: "additional tables, figures, and plots to support and explain your strategic experimental decisions and outcomes"
- **Your report**: Mentions confusion matrices and visualizations, but unclear if they're included in the PDF
- **Status**: Verify all figures are included in the submitted PDF

---

## 📋 **RECOMMENDATIONS**

### High Priority (Should Add)

1. **Task 2 - Noise Addition**
   - Add a sentence: "No noise was added to the dataset, as the original audio features were used as-is to maintain authenticity of the classification task."

2. **Task 3 - Feature Selection/Visualization**
   - Add a sentence or short paragraph: "Before selecting algorithms, exploratory data analysis was performed to understand feature distributions and class imbalance (see Section II). No formal feature selection was performed initially, as all 88 features were used to establish baseline performance."

3. **Task 6 - Hyperparameter Table**
   - Create a formal table showing:
     - Parameter name
     - Values tried
     - Best value found
     - For each model (RF, SVM) and each target (human, asv)

### Medium Priority (Consider Adding)

4. **Task 5 - Category Comparison Table**
   - Consider a table explicitly comparing performance across categories (lamb, goat, sheep, wolf) for both targets

5. **Verify Figures in PDF**
   - Ensure all mentioned visualizations (confusion matrices, performance comparisons) are actually included in the PDF

### Low Priority (Nice to Have)

6. **Task 4 - More Explicit Tradeoffs**
   - Could expand slightly on the tradeoffs of combining systems (already covered but could be more detailed)

---

## ✅ **OVERALL ASSESSMENT**

### Coverage Score: **~95%**

Your report covers almost all requirements comprehensively. The main gaps are:

1. **Explicit statement about noise** (Task 2)
2. **Feature selection/visualization before algorithm selection** (Task 3)
3. **Formal hyperparameter table** (Task 6)

These are minor omissions that can be easily addressed with 1-2 sentences each or a small table.

### Strengths
- ✅ All 7 tasks addressed
- ✅ Line numbers provided for code references
- ✅ Clear structure with sections
- ✅ Comprehensive algorithm justification
- ✅ Good discussion of results and limitations
- ✅ Multiple metrics reported
- ✅ Iterative approach well-documented

### Format
- ✅ Appears to follow IEEE style
- ✅ Within page limit
- ✅ Clear section headings

---

## 🎯 **QUICK FIX CHECKLIST**

Before submission, add:

- [ ] One sentence in Task 2 about not adding noise
- [ ] One sentence in Task 3 about feature selection/visualization before algorithm selection
- [ ] A formal table in Task 6 with hyperparameter ranges and best values
- [ ] Verify all figures are included in the PDF
- [ ] Update line numbers per the mapping document

---

*Generated: [Current Date]*
*Based on comparison of AICE2002 Assignment 1.pdf and report.pdf*

