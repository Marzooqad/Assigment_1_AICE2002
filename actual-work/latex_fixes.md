# LaTeX Fixes for Report

## 1. Multirow Package Issue

**Problem:** `\multirow` command not working

**Solution:** Add this to your preamble (at the top of your LaTeX file, with other `\usepackage` commands):

```latex
\usepackage{multirow}
```

**Full preamble example:**
```latex
\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
% The preceding line is only needed to identify funding in the first footnote. If that is unneeded, please comment it out.
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{multirow}  % <-- ADD THIS LINE
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
```

---

## 2. Image Too Big - Fix Image Sizing

**Problem:** Images are too large and overflow the page

**Solutions:**

### Option A: Scale to text width (Recommended)
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{confusion_matrices_final.png}
\caption{Your caption here}
\label{fig:confusion}
\end{figure}
```

### Option B: Scale to column width (for 2-column format)
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.9\columnwidth]{confusion_matrices_final.png}
\caption{Your caption here}
\label{fig:confusion}
\end{figure}
```

### Option C: Specific width
```latex
\begin{figure}[h]
\centering
\includegraphics[width=7cm]{confusion_matrices_final.png}
\caption{Your caption here}
\label{fig:confusion}
\end{figure}
```

### Option D: Use scale factor
```latex
\begin{figure}[h]
\centering
\includegraphics[scale=0.5]{confusion_matrices_final.png}
\caption{Your caption here}
\label{fig:confusion}
\end{figure}
```

---

## 3. Table with Multirow Example

**If you're using multirow in tables, here's the correct syntax:**

```latex
\begin{table}[h]
\centering
\caption{Your Table Caption}
\label{tab:example}
\begin{tabular}{|c|c|c|c|}
\hline
\multirow{2}{*}{Model} & \multicolumn{2}{c|}{target\_human} & \multirow{2}{*}{target\_asv} \\
\cline{2-3}
 & Accuracy & F1-macro & Accuracy & F1-macro \\
\hline
XGBoost & 0.339 & 0.209 & 0.577 & 0.289 \\
SVM & 0.453 & 0.238 & 0.599 & 0.234 \\
\hline
\end{tabular}
\end{table}
```

**Note:** 
- `\multirow{2}{*}{text}` means: span 2 rows, center alignment, with "text"
- `\multicolumn{2}{c|}{text}` means: span 2 columns, center alignment, with "text"
- `\cline{2-3}` draws a line from column 2 to 3

---

## 4. Complete Figure Example (Fixed)

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{confusion_matrices_final.png}
\caption{Confusion matrices for XGBoost and SVM on both target variables, 
showing per-class prediction patterns. Diagonal elements represent correct 
predictions, while off-diagonal elements show misclassifications.}
\label{fig:confusion}
\end{figure}
```

---

## 5. If Images Still Too Big

**Try these in order:**

1. **Reduce width further:**
   ```latex
   \includegraphics[width=0.7\textwidth]{image.png}
   ```

2. **Use adjustbox package:**
   ```latex
   \usepackage{adjustbox}
   \begin{figure}[h]
   \centering
   \adjustbox{width=0.8\textwidth,center}{\includegraphics{image.png}}
   \caption{Your caption}
   \end{figure}
   ```

3. **Rotate if landscape:**
   ```latex
   \begin{landscape}
   \begin{figure}[h]
   \centering
   \includegraphics[width=0.9\textheight]{image.png}
   \caption{Your caption}
   \end{figure}
   \end{landscape}
   ```
   (Requires `\usepackage{pdflandscape}`)

---

## 6. Quick Checklist

- [ ] Added `\usepackage{multirow}` to preamble
- [ ] Changed all `\includegraphics` to use `width=0.8\textwidth` or similar
- [ ] Tested compilation - no errors
- [ ] Checked that images fit on page
- [ ] Verified multirow tables compile correctly

---

## 7. Common LaTeX Errors and Fixes

**Error: "Undefined control sequence \multirow"**
- **Fix:** Add `\usepackage{multirow}`

**Error: "Overfull \hbox" (image too wide)**
- **Fix:** Reduce `width` parameter (try 0.7 or 0.6)

**Error: "Float too large"**
- **Fix:** Use `[H]` instead of `[h]` (requires `\usepackage{float}`)
- Or use `\resizebox{0.8\textwidth}{!}{\includegraphics{image.png}}`

**Error: "Missing $ inserted"**
- **Fix:** Escape underscores: `target\_human` not `target_human`

---

*Use these fixes to resolve your LaTeX issues!*

