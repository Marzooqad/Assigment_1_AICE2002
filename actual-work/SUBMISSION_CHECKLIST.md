# Submission Checklist
## What to Submit for AICE2002 Assignment 1

---

## Assignment Requirements (from PDF):
- ✅ Submit code (as Python .py scripts or as a Jupyter Notebook)
- ✅ Submit a short report as a PDF (5-6 pages maximum)
- ✅ Submit in **one single zip file** on Moodle
- ✅ Due: November 6, 2025 at 12:00pm (10% late penalty per day)

---

## Files to Include in Your ZIP:

### 1. **REPORT (PDF)**
- ✅ `report.pdf` - Your compiled LaTeX report (NOT the .tex file)
- Make sure it's compiled from `report_with_images.tex`
- Verify it's 5-6 pages (not including references)
- Check all figures are visible and properly sized

### 2. **PYTHON CODE FILES**
Include all your renamed Python files:

- ✅ `step1.py` (data exploration - if you want to include it)
- ✅ `step2 data splitting.py`
- ✅ `step3 preprocessing.py`
- ✅ `step4 model training.py` (baseline)
- ✅ `step4 improved smote.py`
- ✅ `step4 final xgboost.py` (final version)
- ✅ `step4 final xgboost with noise.py` (noise experiment)
- ✅ `step5 hyperparameter tuning.py`
- ✅ `step3_visuals.py` (if you have it)
- ✅ `step5_visuals.py` (if you have it)
- ✅ `visualizations_and_summary.py` (if you have it)

**Note:** You can exclude:
- `step4_final copy.py` (duplicate)
- Any test/temporary files

### 3. **DATA FILES (if required)**
Check if assignment asks for data files. Usually you DON'T submit:
- ❌ `system_Original.csv`
- ❌ `system_X.csv`
- ❌ `system_Y.csv`
- ❌ `train_data.csv`, `valid_data.csv`, `test_data.csv`
- ❌ `.npy` files
- ❌ `.pkl` files

**BUT** if the assignment specifically asks for data, include the original CSV files.

### 4. **OPTIONAL FILES (if helpful)**
- ✅ `requirements.txt` - List of Python packages needed
- ✅ `README.md` - Brief explanation of how to run the code (optional but helpful)

---

## Pre-Submission Checklist:

### Report (PDF):
- [ ] Compiled from `report_with_images.tex`
- [ ] All 5 figures are visible and properly sized
- [ ] Table I is formatted correctly (multirow fixed)
- [ ] All line number references are correct (check `line_number_mapping.md`)
- [ ] All 7 tasks are clearly addressed
- [ ] References section includes all code files
- [ ] Page count is 5-6 pages (excluding references)
- [ ] No LaTeX compilation errors
- [ ] PDF opens and displays correctly

### Code Files:
- [ ] All files are renamed correctly:
  - `step2 data splitting.py` (not `step2.py`)
  - `step3 preprocessing.py` (not `step3.py`)
  - `step4 model training.py` (not `step4.py`)
  - `step4 improved smote.py` (not `step4_improv.py`)
  - `step4 final xgboost.py` (not `step4_final.py`)
  - `step5 hyperparameter tuning.py` (not `step5.py`)
- [ ] All code files run without errors
- [ ] Code is readable and has comments
- [ ] No duplicate files included

### File Organization:
- [ ] Create a folder structure:
  ```
  assignment_submission/
  ├── report.pdf
  ├── code/
  │   ├── step2 data splitting.py
  │   ├── step3 preprocessing.py
  │   ├── step4 model training.py
  │   ├── step4 improved smote.py
  │   ├── step4 final xgboost.py
  │   ├── step4 final xgboost with noise.py
  │   └── step5 hyperparameter tuning.py
  ├── requirements.txt (optional)
  └── README.md (optional)
  ```

---

## How to Create the ZIP File:

### Option 1: Manual Selection
1. Create a folder called `assignment_submission`
2. Copy all required files into it
3. Right-click → "Send to" → "Compressed (zipped) folder"
4. Name it: `[YourStudentID]_AICE2002_Assignment1.zip`

### Option 2: Using Command Line
```powershell
# Create folder
New-Item -ItemType Directory -Path "assignment_submission"

# Copy files
Copy-Item "report.pdf" -Destination "assignment_submission\"
Copy-Item "step2 data splitting.py" -Destination "assignment_submission\code\"
Copy-Item "step3 preprocessing.py" -Destination "assignment_submission\code\"
# ... (copy all other files)

# Create ZIP
Compress-Archive -Path "assignment_submission\*" -DestinationPath "[YourStudentID]_AICE2002_Assignment1.zip"
```

---

## Final Verification:

Before submitting, verify:

1. **ZIP file opens correctly**
2. **All Python files are included and have correct names**
3. **Report PDF is included (not .tex file)**
4. **Report PDF opens and all figures are visible**
5. **File size is reasonable** (should be < 50MB, likely < 10MB)
6. **No personal information** (except student ID which should be in report)

---

## Submission Format:

**ZIP File Name:** `[36517445]_AICE2002_Assignment1.zip`

**Contents:**
```
[36517445]_AICE2002_Assignment1.zip
├── report.pdf
├── step2 data splitting.py
├── step3 preprocessing.py
├── step4 model training.py
├── step4 improved smote.py
├── step4 final xgboost.py
├── step4 final xgboost with noise.py
└── step5 hyperparameter tuning.py
```

---

## Quick Submission Steps:

1. ✅ Compile `report_with_images.tex` → `report.pdf`
2. ✅ Verify PDF has all figures and is 5-6 pages
3. ✅ Collect all Python files with correct names
4. ✅ Create ZIP file with report.pdf + all .py files
5. ✅ Test ZIP opens correctly
6. ✅ Upload to Moodle before deadline

---

## Important Notes:

- ⚠️ **Submit PDF, NOT LaTeX source** (unless assignment specifically asks for .tex)
- ⚠️ **Check file names match report references** (use renamed files)
- ⚠️ **Verify line numbers in report match actual code** (see `line_number_mapping.md`)
- ⚠️ **Don't include data files** unless specifically required
- ⚠️ **Don't include output files** (.csv, .png, .pkl, .npy) unless required

---

**Good luck with your submission! 🎓**

