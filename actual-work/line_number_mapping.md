# Line Number Mapping Document
## File Naming and Code Reference Updates

This document maps the line number references in your report to the actual line numbers in your renamed Python files.

---

## File Renaming Summary

| Old File Name | New File Name | Status |
|--------------|--------------|--------|
| `step2.py` | `step2 data splitting.py` | ✅ Renamed |
| `step3.py` | `step3 preprocessing.py` | ✅ Renamed |
| `step4.py` | `step4 model training.py` | ✅ Renamed |
| `step4_improv.py` | `step4 improved smote.py` | ✅ Renamed |
| `step4_final.py` | `step4 final xgboost.py` | ✅ Renamed |
| `step5.py` | `step5 hyperparameter tuning.py` | ✅ Renamed |

---

## Line Number Reference Mapping

### 1. step2 data splitting.py

**Report References:**
- Lines 61-100: Speaker-based splitting implementation
- Lines 73-85: Stratification by gender
- Lines 131-180: Class distribution analysis

**Actual Line Numbers:**
- ✅ Lines 61-100: Speaker-based splitting implementation (MATCHES)
- ⚠️ Lines 69-74: Stratification by gender (CLOSE - report says 73-85, actual is 69-74)
- ❌ Lines 108-147: Class distribution analysis (DIFFERENT - report says 131-180, actual is 108-147)

**Update Required:**
- Change "Lines 73-85" → "Lines 69-74" (stratification)
- Change "Lines 131-180" → "Lines 108-147" (class distribution)

---

### 2. step3 preprocessing.py

**Report References:**
- Lines 121-180: Feature scaling with StandardScaler

**Actual Line Numbers:**
- ❌ Lines 69-143: Feature scaling implementation (DIFFERENT - report says 121-180, actual is 69-143)
  - Lines 69-79: StandardScaler initialization and transformation
  - Lines 81-86: Scaling statistics and verification
  - Lines 88-115: Class distribution analysis (part of preprocessing)

**Update Required:**
- Change "Lines 121-180" → "Lines 69-143" (feature scaling)

---

### 3. step4 model training.py

**Report References:**
- Lines 80-105: Baseline Random Forest and SVM implementation with class weighting

**Actual Line Numbers:**
- ❌ Lines 37-52: Classifier initialization with class_weight='balanced' (DIFFERENT)
- ❌ Lines 60-123: train_and_evaluate function (DIFFERENT - report says 80-105, actual spans 60-123)

**Update Required:**
- Change "Lines 80-105" → "Lines 37-52" (classifier initialization) and "Lines 60-123" (training function)
- OR combine: "Lines 37-123" (baseline implementation)

---

### 4. step4 improved smote.py

**Report References:**
- Lines 110-120: SMOTE oversampling implementation
- Lines 80-100: Aggressive class weighting

**Actual Line Numbers:**
- ❌ Lines 38-70: Aggressive class weights definition (DIFFERENT - report says 80-100, actual is 38-70)
  - Lines 38-53: Class weights for target_human
  - Lines 55-70: Class weights for target_asv
- ❌ Lines 89-91: SMOTE implementation (DIFFERENT - report says 110-120, actual is 89-91)
  - Line 90: SMOTE initialization
  - Line 91: SMOTE fit_resample

**Update Required:**
- Change "Lines 80-100" → "Lines 38-70" (aggressive class weighting)
- Change "Lines 110-120" → "Lines 89-91" (SMOTE implementation)

---

### 5. step4 final xgboost.py

**Report References:**
- Lines 90-110: XGBoost implementation with extreme weighting
- Lines 170-200: Label encoding for multiclass support

**Actual Line Numbers:**
- ⚠️ Lines 89-177: train_and_evaluate function (PARTIALLY MATCHES)
  - Lines 89-110: Function definition and SMOTE (MATCHES 90-110)
  - Lines 110-129: Label encoding for XGBoost (DIFFERENT - report says 170-200, actual is 110-129)
  - Lines 131-177: Model training and evaluation
- ⚠️ Lines 179-200: Model training calls (MATCHES 170-200 range)

**Update Required:**
- Change "Lines 90-110" → "Lines 89-110" (XGBoost with extreme weighting - minor adjustment)
- Change "Lines 170-200" → "Lines 110-129" (label encoding) and "Lines 179-200" (model training)

---

### 6. step5 hyperparameter tuning.py

**Report References:**
- Lines 101-250: GridSearchCV implementation with 3-fold cross-validation

**Actual Line Numbers:**
- ❌ Lines 66-108: tune_and_evaluate function (DIFFERENT - report says 101-250, actual starts at 66)
  - Lines 72-80: GridSearchCV setup
  - Lines 83-89: Grid search execution
- ❌ Lines 110-162: Hyperparameter tuning calls (DIFFERENT - report says 101-250, actual is 110-162)
- ❌ Lines 177-227: Results processing and saving (DIFFERENT)

**Update Required:**
- Change "Lines 101-250" → "Lines 66-108" (tune_and_evaluate function) and "Lines 110-227" (full implementation)

---

## Summary of Required Report Updates

### Section II (Quantifying Bias)
- **Line 51**: Change "Lines 131-180 of step2 data splitting.py" → "Lines 108-147 of step2 data splitting.py"
- **Line 82**: Change "Lines 73-85, step2 data splitting.py" → "Lines 69-74, step2 data splitting.py"

### Section III (Task 1)
- **Line 95**: Change "Lines 61-100 of step2 data splitting.py" → "Lines 61-100 of step2 data splitting.py" ✅ (No change needed)

### Section IV (Task 2)
- **Line 123**: Change "Lines 121-180 in step3 preprocessing.py" → "Lines 69-143 in step3 preprocessing.py"

### Section V (Task 3)
- **Line 196**: Change "Lines 80-95 in step4 model training.py" → "Lines 37-52 in step4 model training.py" (classifier initialization)
- **Line 217**: Change "Lines 96-105 in step4 model training.py" → "Lines 60-123 in step4 model training.py" (SVM implementation)

### Section IV (Task 2 - Continued)
- **Line 139**: Change "Lines 110-120 of step4 improved smote.py" → "Lines 89-91 of step4 improved smote.py"
- **Line 156**: Change "Lines 80-100 in step4 improved combined.py" → "Lines 38-70 in step4 improved smote.py"

### Section V (Task 3 - Continued)
- **Line 181**: Change "step4 final xgboost.py with these settings" → "step4 final xgboost.py (Lines 47-81) with these settings"
- **Line 431**: Change "Lines 90-110, 170-200" → "Lines 89-177" (full implementation)

### Section VIII (Task 6)
- **Line 316**: Change "step5 hyperparameter tuning.py" → "step5 hyperparameter tuning.py (Lines 66-227)"

### References Section
- **[4]**: Change "Lines 61-100" → "Lines 61-100" ✅ (No change)
- **[4]**: Change "Lines 131-180" → "Lines 108-147"
- **[5]**: Change "Lines 121-180" → "Lines 69-143"
- **[6]**: Change "Lines 80-105" → "Lines 37-123"
- **[7]**: Change "Lines 110-120" → "Lines 89-91"
- **[7]**: Change "Lines 80-100" → "Lines 38-70"
- **[8]**: Change "Lines 90-110, 170-200" → "Lines 89-177"
- **[9]**: Change "Lines 101-250" → "Lines 66-227"

---

## Notes

1. **Most line numbers have shifted** because:
   - Code was refactored or reorganized
   - Comments and blank lines were added/removed
   - Functions were restructured

2. **Some references are close** (within 10-20 lines), suggesting minor edits were made.

3. **The file names in the report should now match** the renamed files exactly.

4. **Recommendation**: Update all line number references in the report PDF to match the actual line numbers listed above.

---

## Quick Reference Table

| File | Report Lines | Actual Lines | Status |
|------|-------------|--------------|--------|
| step2 data splitting.py - Speaker splitting | 61-100 | 61-100 | ✅ Match |
| step2 data splitting.py - Stratification | 73-85 | 69-74 | ⚠️ Close |
| step2 data splitting.py - Class distribution | 131-180 | 108-147 | ❌ Different |
| step3 preprocessing.py - Feature scaling | 121-180 | 69-143 | ❌ Different |
| step4 model training.py - Baseline | 80-105 | 37-123 | ❌ Different |
| step4 improved smote.py - SMOTE | 110-120 | 89-91 | ❌ Different |
| step4 improved smote.py - Weights | 80-100 | 38-70 | ❌ Different |
| step4 final xgboost.py - XGBoost | 90-110 | 89-110 | ⚠️ Close |
| step4 final xgboost.py - Encoding | 170-200 | 110-129, 179-200 | ❌ Different |
| step5 hyperparameter tuning.py | 101-250 | 66-227 | ❌ Different |

---

*Generated: [Current Date]*
*All line numbers verified against actual source files*

