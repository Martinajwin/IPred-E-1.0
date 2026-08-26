# Inhibitor Predictor for EGFR (IPRED-E 1.0)

### Overview
IPRED-E 1.0 is a Streamlit web application implementing a rigorous, sequential machine learning pipeline to screen and predict Epidermal Growth Factor Receptor (EGFR) inhibitors. It employs 1D and 2D topological Mordred molecular descriptors alongside a highly calibrated Support Vector Machine (SVM) classifier and an XGBoost Regressor. The pipeline is extensively validated on external test sets and decoy datasets to ensure robust classification of active molecules and reliable prediction of their binding affinity (pIC50).

### Features
* **Flexible Input:** Input SMILES strings manually or via CSV upload.
* **Automated Feature Extraction:** Automatically computes the specific subsets of topological descriptors required for both models.
* **Sequential Prediction Workflow:** First acts as a gatekeeper by separating Active from Inactive compounds (SVM), then predicts the exact potency (pIC50) of the active hits (XGBoost).
* **Strict Applicability Domain (AD) Logic:** Implements independent k-NN based AD constraints for *both* the classification and regression models to flag structural extrapolations and actively prevent decoys or reactive artifacts from advancing.
* **Integrated Diagnostics & Export:** Download standard screening results or detailed AD spatial diagnostic reports for further analysis.

---

### Access the Web Tool
You can access and use the IPRED-E 1.0 virtual screening pipeline directly through your web browser without any installation:

🔗 **[Launch IPRED-E 1.0 Web Tool Here]([https://ipred-e-1-clf-reg-screening.streamlit.app/])**

---

### Citation
If you utilize the IPRED-E 1.0 webtool or its concepts in your research, please cite:

> **IPRED-E 1.0 Webtool** | D. Kumar, A. J. Martin | Version 1.0 (2026).
> **Webtool URL:** *(https://ipred-e-1-clf-reg-screening.streamlit.app/)* 

---

### Copyright & Intellectual Property

**© 2026 Manipal Academy of Higher Education (MAHE). All rights reserved.**

**Authors/Creators:** Dileep Kumar and Ajwin Joseph Martin

The source code, algorithms, sequential consensus logic, and trained models associated with IPRED-E 1.0 are the exclusive intellectual property of Manipal Academy of Higher Education (MAHE). This repository is hosted on the creators' personal account and made public for the sole purpose of deploying the Streamlit web application and facilitating transparency for academic peer review.

**Permissions:**
* You are permitted to view the source code for educational and peer-review purposes.
* You are permitted to use the deployed web tool via the provided Streamlit URL for your own virtual screening tasks, provided proper citation is given.

**Restrictions:**
* You may **NOT** copy, reproduce, distribute, modify, or create derivative works from this codebase.
* You may **NOT** use the code or models for any commercial or private non-commercial deployment without explicit written permission from the copyright owner (MAHE) and the authors.

For licensing inquiries or permission requests, please contact the authors directly.
