# LinkedIn AI Slop Detector 🚀

A lightning-fast Google Chrome extension that instantly detects and highlights AI-generated "Slop" (Hustle Culture, Marketing Shills, etc.) in your LinkedIn feed as you scroll.

Built using an ensemble of a **Fine-Tuned Laya Model** (System 1 Decision Engine) and robust DOM structural parsing.

## How it Works

Unlike traditional AI wrappers that use generative LLMs (like GPT-4 or Claude) which suffer from high latency, API costs, and hallucinations, this extension uses **Laya** by ConvAI Innovations.

Laya is an open-source (Apache 2.0) non-autoregressive decision engine. It evaluates text in a single forward pass (~33ms on GPU) to return calibrated probabilities without ever generating text.

1. **DOM TreeWalker:** The extension bypasses LinkedIn's heavily obfuscated CSS classes by utilizing a structural DOM `TreeWalker` to extract post text reliably.
2. **Local Heuristics:** It performs a localized, ultra-fast regex check (e.g., emoji density) to catch obvious spam instantly.
3. **Laya ML Inference:** For nuanced semantic checks, it queries the fine-tuned Laya backend.
4. **Ensemble Scoring:** The badge updates in real-time right inside your feed.

## The Model: Laya vs Jev

While recent proprietary APIs like TypeSafe's **Jev** popularized non-autoregressive structured decisions, **Laya** (built by Nandakishor) offers a superior architecture:
* **Cost:** $0.00 (100% open-source) vs $0.042/1M tokens.
* **Speed:** 32.8ms execution vs 150ms+.
* **Hallucination:** Zero. It outputs pure schema probabilities.

## Installation

1. Clone or download this repository.
2. Open Google Chrome and go to `chrome://extensions/`.
3. Enable **Developer mode** in the top right corner.
4. Click **Load unpacked** and select the `extension/` directory.
5. Refresh your LinkedIn feed!

## Local Development & Training

If you want to modify the dataset or retrain the Laya model yourself:

1. **Set up a Python virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. **Install dependencies:**
   ```bash
   pip install modal pandas datasets scikit-learn transformers
   ```
3. **Generate the Synthetic Dataset:**
   ```bash
   python build_dataset.py
   ```
   *This will generate `laya_slop_dataset.csv` with over 11,000 labeled examples.*

4. **Fine-tune the Model on Modal (GPU):**
   ```bash
   modal run finetune_laya.py
   ```
   *This spins up a remote T4 GPU, trains the model for 1 epoch, and permanently saves it to a Modal Volume.*

## Backend Setup (Modal)

To host the Laya model yourself:
1. Install Modal: `pip install modal`
2. Authenticate: `modal token new`
3. Deploy the inference API: `modal deploy serve_laya.py`
4. Update the `LAYA_API_URL` in `content.js` to your deployed endpoint.

## Dataset
The model was fine-tuned on a custom dataset combining HuggingFace benchmarks (`NicolaiSivesind/AI-generated-vs-Human-written`) with thousands of synthetically generated LinkedIn "Hustle Culture" and "B2B Marketing" templates.
