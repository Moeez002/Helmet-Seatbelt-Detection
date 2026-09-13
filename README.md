# Helmet & Seatbelt Detection

YOLO-based Streamlit app for detecting helmet usage and seatbelt usage in uploaded images.

🔗 **Live demo:** _add your Streamlit Cloud URL here after deploying_

## Project structure

```
helmet-seatbelt-detection/
├── app.py                  # Streamlit app (this is what gets deployed)
├── requirements.txt
├── models/
│   ├── helmet_model.pt     # trained YOLOv8 weights — see note below
│   └── seatbelt_model.pt
└── README.md
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model weights

`models/helmet_model.pt` and `models/seatbelt_model.pt` are trained YOLOv8 checkpoints
(not included in git if they exceed GitHub's size limits — see below).

- If each file is **under ~100MB**, just add them to the repo normally.
- If either is **larger**, use [Git LFS](https://git-lfs.com/):
  ```bash
  git lfs install
  git lfs track "*.pt"
  git add .gitattributes
  ```
- Alternatively, host the weights on a GitHub Release or Hugging Face Hub and
  download them at startup instead of committing them.

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Select this repo, branch `main`, main file path `app.py`.
4. Deploy. You'll get a permanent public URL viewers can open directly.

Note: the free tier has ~1GB RAM and no GPU, so inference is slower than on Colab,
and the app cold-starts after periods of inactivity.

## Training

The model training code (dataset prep, class balancing, YOLO `.train()` calls) lives
separately in the original notebook and is not part of the deployed app.
