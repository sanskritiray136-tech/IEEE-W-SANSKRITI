# Student Success Copilot 🎓

A polished, interactive machine-learning web app that helps turn student performance patterns into practical, supportive next steps.

## Why this is useful
Instead of only showing a pass/fail prediction, Student Success Copilot combines a model estimate with a short, supportive action plan. The UI includes a student profile form, model comparison chart, estimated pass likelihood, tailored suggestions, and a downloadable CSV snapshot.

**Important:** This is an educational prototype. It must not be used to label students or make high-stakes decisions about real students.

## Features
- Attractive responsive Streamlit interface
- Student profile form organized into tabs
- Two ML models: Random Forest and K-Nearest Neighbors
- Test-set accuracy comparison
- Pass-likelihood estimate for a sample profile
- Supportive, rule-based next-step suggestions
- Downloadable support snapshot CSV
- Excludes `G1`, `G2`, and `G3` from model inputs to avoid target leakage

## Dataset
UCI Student Performance, mathematics course (`student-mat.csv`).

- Original dataset page: https://archive.ics.uci.edu/dataset/320/student+performance
- CSV source used by this app: https://raw.githubusercontent.com/uciml/student-alcohol-consumption/master/student-mat.csv

The app downloads the data on startup, so an internet connection is required.

## Run it on your computer
1. Install Python 3.9 or later.
2. Extract this project folder.
3. Open a terminal in the folder.
4. Create and activate a virtual environment (optional but recommended):

   **Windows**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Start the app:
   ```bash
   streamlit run app.py
   ```
7. Streamlit will print a local address, usually `http://localhost:8501`. Open it in your browser.

## Publish it online for a shareable link
To make it open for others without running it locally:

1. Create a GitHub repository, for example `student-success-copilot`.
2. Upload `app.py`, `requirements.txt`, and this `README.md`.
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
5. Deploy. Wait for the build to finish; Streamlit will provide a public URL.
6. Open that URL in a private/incognito browser window to test it.

The public URL only exists after you deploy it from your account. This project package does not itself create a hosted website.

## Model and target
A pass is defined as final mathematics grade `G3 >= 10` on a 0–20 scale; otherwise, the label is fail. The models are trained on an 80/20 stratified split with a fixed random seed. The app displays measured accuracy values from that run.

## Responsible use and limitations
- This dataset is small and specific to a particular context.
- Accuracy on one split does not prove the model generalizes to another school.
- Demographic and family variables may encode historical inequities.
- The estimate is not a certainty, diagnosis, or judgement of a student's ability.
- Use suggestions to start supportive conversations, not to restrict opportunities.
