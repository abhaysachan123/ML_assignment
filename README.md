# Maximum Likelihood Estimation for Regression

**Student:** Abhay Sachan  
**Batch:** 22  
**Unit:** V  
**Topic No.:** 16  
**Semester:** III (Aug–Dec 2026)  
**Date:** October 16, 2026  
**Course:** Applied Machine Learning (CSAI2017P)  
**Topic:** Maximum Likelihood Estimation for Linear Regression with Gaussian Errors

## Files and folders

- `MLE.pdf` — compiled report.
- `MLE.tex` — LaTeX source used to compile the report.
- `MLE.txt` — plain-text copy of the LaTeX source.
- `MLE.py` — reproducible Python implementation.
- `figures/fit.png` — observed synthetic data and fitted regression line.
- `figures/likelihood.png` — profile negative log-likelihood plot.
- `figures/frame1.png` through `figures/frame4.png` — static demonstration frames used in the report.
- `MLE_animation.gif` — animated candidate regression line, stored independently in the project root.
- `results.txt` — output values from the demonstration.
- `requirements.txt` — Python package dependencies.

## Run the demonstration

Install the dependencies and run the script from this folder:

```bash
python -m pip install -r requirements.txt
python MLE.py
```

The script regenerates static plots in `figures/` and saves `MLE_animation.gif` in the project root. The example uses a fixed random seed, so the synthetic dataset and estimates are reproducible for the same compatible numerical-library environment.

## Open or edit the report in Overleaf

Upload the project ZIP using **New Project → Upload Project**. Open `MLE.tex` and compile. If Overleaf expects `main.tex`, either select `MLE.tex` as the main document in the project settings or rename it to `main.tex` there.

## Before submission

- Confirm the student details and date.
- Read the derivation and make sure you can explain the Gaussian-error assumption, the coefficient estimates, and why the variance MLE is RSS divided by `n`.
- Review all figures and the compiled PDF.
- Complete the AI-tool declaration honestly and in accordance with the course policy; the draft identifies the kinds of assistance used, but the student must fill in the approximate share required by the faculty.
