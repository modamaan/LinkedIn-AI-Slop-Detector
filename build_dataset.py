"""
LinkedIn AI Slop Dataset Builder v3
=====================================
Strategy:
  1. Generate synthetic LinkedIn slop (no API needed, instant)
  2. HuggingFace: NicolaiSivesind/AI-generated-vs-Human-written (works)
  3. HuggingFace: webis/tldr-17 (Reddit human summaries, works)
  4. Reddit via old.reddit.com with session cookies (best effort)
"""

import random
import time
import sys
import requests
import pandas as pd
from pathlib import Path

OUTPUT_FILE = "laya_slop_dataset.csv"
random.seed(42)


# ══════════════════════════════════════════════════════════════════════════
# SOURCE 1: Synthetic LinkedIn Slop Generator (label=1)
# No API, no internet, instant, perfect labels
# ══════════════════════════════════════════════════════════════════════════

SITUATIONS = [
    "fired from my dream job", "rejected by 47 companies",
    "told I wasn't good enough", "broke with $200 in my account",
    "sleeping on my friend's couch", "laid off after 6 years",
    "passed over for promotion again", "failed my third startup",
    "ghosted by every investor I pitched", "burned out and exhausted",
]

ACHIEVEMENTS = [
    "lead a team of 50 people", "run a 7-figure business",
    "got hired by a FAANG company", "launched a product used by millions",
    "speak at global conferences", "mentor 1000+ professionals",
    "work 4 hours a day from anywhere in the world",
    "turned down a $2M acquisition", "built the life I always dreamed of",
    "wake up excited about Mondays",
]

LESSONS = [
    ["Rejection is redirection, not failure",
     "Your worst day builds your best character",
     "The universe always has a plan"],
    ["Consistency beats talent every time",
     "Your network is your net worth",
     "Discomfort is where growth lives"],
    ["Stop waiting for permission to start",
     "Done is better than perfect",
     "Every expert was once a beginner"],
    ["Invest in yourself before anything else",
     "The only failure is not trying",
     "Success leaves clues — find them"],
    ["Surround yourself with people who challenge you",
     "Your mindset is your greatest asset",
     "Gratitude unlocks abundance"],
]

ENDINGS = [
    "What's your take? Drop a comment below {emoji}",
    "Has this ever happened to you? I'd love to hear your story {emoji}",
    "Tag someone who needs to hear this today {emoji}",
    "Follow for more lessons on resilience and growth {emoji}",
    "Agree or disagree? Let me know below {emoji}",
    "Save this post for when you need a reminder {emoji}",
    "Which point resonates most with you? {emoji}",
]

EMOJIS = ["👇", "💪", "🚀", "✨", "🙏", "💡", "🔥"]

OPENERS = [
    "{n} years ago, I was {situation}.\n\nToday, I {achievement}.\n\nHere's what I learned:\n\n",
    "I was {situation}.\n\nEveryone told me to give up.\n\nI didn't listen.\n\nNow I {achievement}.\n\n{n} lessons that changed everything:\n\n",
    "Unpopular opinion: Being {situation} was the BEST thing that ever happened to me.\n\nHere's why:\n\n",
    "Nobody talks about what it's really like to be {situation}.\n\nSo I will.\n\nAfter {n} years, here's what I wish someone told me:\n\n",
    "I went from {situation} to {achievement} in {n} years.\n\nPeople ask me how. The answer is simple:\n\n",
]

def generate_slop_post() -> str:
    n = random.randint(1, 7)
    situation = random.choice(SITUATIONS)
    achievement = random.choice(ACHIEVEMENTS)
    lessons = random.choice(LESSONS)
    opener = random.choice(OPENERS).format(n=n, situation=situation, achievement=achievement)
    
    numbered = ""
    for i, lesson in enumerate(lessons, 1):
        numbered += f"{i}. {lesson}\n"
    
    ending = random.choice(ENDINGS).format(emoji=random.choice(EMOJIS))
    return opener + numbered + "\n" + ending


def build_synthetic_slop(n: int = 3000) -> list:
    print(f"\n[GEN] Generating {n} synthetic LinkedIn slop posts...")
    posts = []
    for i in range(n):
        text = generate_slop_post()
        # Add variation: sometimes add emojis in bullets
        if random.random() > 0.5:
            text = text.replace("1.", "1. ✅").replace("2.", "2. ✅").replace("3.", "3. ✅")
        posts.append({"text": text, "label": 1, "source": "synthetic/linkedin-slop", "score": 0})
    print(f"  [OK] Generated {len(posts)} slop examples")
    return posts


# ══════════════════════════════════════════════════════════════════════════
# SOURCE 2: HuggingFace datasets (working ones)
# ══════════════════════════════════════════════════════════════════════════

def load_hf_datasets() -> list:
    results = []

    # ── Dataset A: NicolaiSivesind/AI-generated-vs-Human-written ─────────
    # This is a clean Parquet dataset, no loading scripts
    try:
        print("\n[GET] HuggingFace: AI-generated-vs-Human-written ...")
        from datasets import load_dataset
        ds = load_dataset("NicolaiSivesind/AI-generated-vs-Human-written", split="train")
        
        # Inspect first row to find column names
        first = ds[0]
        print(f"  Columns: {list(first.keys())}")
        
        count = 0
        for row in ds:
            # Try common column name patterns
            text = (row.get("text") or row.get("content") or row.get("essay") or
                    row.get("Text") or "")
            lbl_raw = (row.get("generated") or row.get("label") or
                       row.get("is_generated") or row.get("Label") or None)
            
            if not text or len(str(text)) < 80:
                continue
            if lbl_raw is None:
                continue
            
            # Normalize label to 0/1
            try:
                lbl = int(lbl_raw)
            except (ValueError, TypeError):
                lbl = 1 if str(lbl_raw).lower() in ("ai", "generated", "1", "true") else 0
            
            results.append({
                "text": str(text)[:2000],
                "label": lbl,
                "source": "hf/ai-vs-human-written",
                "score": 0
            })
            count += 1
            if count >= 8000:
                break
        
        print(f"  [OK] {count} examples loaded")

    except Exception as e:
        print(f"  [SKIP] AI-vs-Human-written: {e}")

    # ── Dataset B: webis/tldr-17 (human Reddit summaries → label=0) ──────
    try:
        print("\n[GET] HuggingFace: webis/tldr-17 (human writing) ...")
        from datasets import load_dataset
        ds = load_dataset("webis/tldr-17", split="train", trust_remote_code=False)
        
        first = ds[0]
        print(f"  Columns: {list(first.keys())}")
        
        count = 0
        for row in ds:
            text = row.get("content") or row.get("summary") or row.get("text") or ""
            if len(str(text)) < 80:
                continue
            results.append({
                "text": str(text)[:2000],
                "label": 0,  # human-written Reddit posts
                "source": "hf/tldr17-human",
                "score": 0
            })
            count += 1
            if count >= 5000:
                break
        
        print(f"  [OK] {count} human examples from tldr-17")

    except Exception as e:
        print(f"  [SKIP] tldr-17: {e}")

    # ── Dataset C: inspect artem9k properly ──────────────────────────────
    try:
        print("\n[GET] HuggingFace: ai-text-detection-pile (inspecting columns) ...")
        from datasets import load_dataset
        ds = load_dataset("artem9k/ai-text-detection-pile", split="train")
        
        first = ds[0]
        print(f"  Columns: {list(first.keys())}")
        print(f"  Sample row: { {k: str(v)[:60] for k, v in first.items()} }")

        count = 0
        for row in ds:
            # Based on actual column inspection
            text = ""
            lbl = None
            for tk in ("text", "content", "sample", "Text"):
                if row.get(tk):
                    text = str(row[tk])
                    break
            for lk in ("label", "generated", "is_ai", "source", "Label"):
                if row.get(lk) is not None:
                    lbl = row[lk]
                    break

            if len(text) < 80 or lbl is None:
                continue

            try:
                lbl_int = int(lbl)
            except (ValueError, TypeError):
                lbl_int = 1 if str(lbl).lower() in ("ai", "generated", "1", "gpt", "chatgpt") else 0

            results.append({
                "text": text[:2000],
                "label": lbl_int,
                "source": "hf/ai-detection-pile",
                "score": 0
            })
            count += 1
            if count >= 8000:
                break

        print(f"  [OK] {count} examples from ai-detection-pile")

    except Exception as e:
        print(f"  [SKIP] ai-detection-pile: {e}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# SOURCE 3: Reddit (best effort, graceful fallback)
# ══════════════════════════════════════════════════════════════════════════

def try_reddit(subreddit: str, label: int, pages: int = 5) -> list:
    posts = []
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    })

    print(f"\n[GET] Reddit r/{subreddit} (best effort)...")
    after = None

    for page in range(pages):
        url = f"https://www.reddit.com/r/{subreddit}/top.json?limit=100&t=all"
        if after:
            url += f"&after={after}"
        try:
            r = session.get(url, timeout=10)
            if r.status_code != 200:
                print(f"  [SKIP] HTTP {r.status_code} - Reddit blocking us, skipping")
                break
            data = r.json()["data"]
            after = data.get("after")
            for p in data["children"]:
                d = p["data"]
                t = d.get("selftext", "").strip()
                if len(t) > 80 and t not in ("[deleted]", "[removed]"):
                    posts.append({"text": t[:2000], "label": label,
                                  "source": f"r/{subreddit}", "score": d.get("score", 0)})
            print(f"  Page {page+1}: {len(posts)} posts total")
            if not after:
                break
            time.sleep(2)
        except Exception as e:
            print(f"  [SKIP] {e}")
            break

    return posts


# ══════════════════════════════════════════════════════════════════════════
# MAIN: Assemble + Save
# ══════════════════════════════════════════════════════════════════════════

def build_dataset():
    all_data = []

    # 1. Synthetic slop — guaranteed, instant
    all_data.extend(build_synthetic_slop(n=3000))

    # 2. HuggingFace datasets
    all_data.extend(load_hf_datasets())

    # 3. Reddit (best effort, might fail with 403)
    all_data.extend(try_reddit("LinkedInLunatics", label=1, pages=5))
    all_data.extend(try_reddit("cscareerquestions", label=0, pages=3))
    all_data.extend(try_reddit("careerguidance", label=0, pages=3))

    if not all_data:
        print("\n[ERROR] No data at all — check internet connection")
        sys.exit(1)

    df = pd.DataFrame(all_data)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 80]
    df = df.drop_duplicates(subset="text")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n" + "="*55)
    print("FINAL DATASET STATS")
    print("="*55)
    print(f"Total   : {len(df)}")
    print(f"Slop =1 : {(df.label == 1).sum()}")
    print(f"Human=0 : {(df.label == 0).sum()}")
    print("\nBy source:")
    print(df.groupby(["source", "label"]).size().to_string())

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n[SAVED] -> {OUTPUT_FILE}  ({len(df)} rows)")

    print("\n--- SLOP SAMPLE ---")
    s = df[df.label == 1].sample(1).iloc[0]
    print(f"Source: {s['source']}")
    print(s["text"][:400])

    print("\n--- HUMAN SAMPLE ---")
    h = df[df.label == 0].sample(1).iloc[0]
    print(f"Source: {h['source']}")
    print(h["text"][:400])

    return df


if __name__ == "__main__":
    build_dataset()
