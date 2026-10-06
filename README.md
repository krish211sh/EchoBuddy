# EchoBuddy

## 📌 Project Overview

EchoBuddy is an AI companion for students and children. It chats like a friendly buddy, listens to everyday problems, detects early signs of stress, anxiety or sadness using deep learning, and gives gentle encouragement using motivational stories and teachings from the Bhagavad Gita, the Ramayana and Buddhist texts (Dhammapada). A parent dashboard shows a short wellbeing summary, never the full conversation.

> ⚠️ EchoBuddy is an **early-signal support tool, not a medical or diagnostic system.** It does not replace parents, teachers or mental-health professionals.

## 🎯 Objectives

- Build a friendly chat assistant for students.
- Detect emotions in messages using a fine-tuned DistilBERT model.
- Track mood over time and flag a rising trend of concern.
- Generate a simple summary for parents on a dashboard.
- Teach positivity and moral values with Sanskrit/Pali verses explained in simple English, and real-life success stories.
- Follow privacy and ethical practices for children's data.

## 📊 Dataset

- **GoEmotions** (Google Research): about 58,000 Reddit comments labelled with 27 emotions plus neutral. Loaded from Hugging Face: `google-research-datasets/go_emotions` (simplified version).
- The 28 emotion labels are grouped into **sad, anxiety, anger, happy** (see `emotion_map.py`).
- The wisdom library (31 items) is hand-written: Gita verses with Sanskrit text and English meaning, Dhammapada verses, Ramayana stories, real-life stories and calming tips.

## 🛠️ Technologies Used

- Python, PyTorch
- Hugging Face Transformers and Datasets (DistilBERT)
- Scikit-learn, NumPy
- FastAPI, Uvicorn
- HTML, CSS, JavaScript (chat and parent dashboard)

## 🔄 Project Workflow

1. Load GoEmotions and tokenize text (max length 64).
2. Fine-tune DistilBERT for multi-label emotion classification (`train.py`).
3. Evaluate with F1 (micro and macro), precision and recall.
4. Group emotions into wellbeing categories (`emotion_map.py`).
5. Train a small LSTM on mood-score sequences to flag concerning trends (`trend_lstm.py`, proof of concept on synthetic data).
6. Serve predictions through a FastAPI endpoint (`server.py`).
7. The web app (`index.html`) shows the chat, the parent dashboard and the wisdom library. If the API is not running, it automatically falls back to keyword-based detection.

## 🤖 Machine Learning Models

### DistilBERT (emotion classifier)
A smaller, faster version of BERT, fine-tuned for multi-label classification with a sigmoid output and a 0.3 threshold.

### LSTM (mood trend)
A small recurrent network that reads the last 10 emotion-score vectors and outputs a concern probability. Currently trained on synthetic sequences only, as a proof of concept.

### Safety layer
A rule that always triggers an alert for serious statements (for example about self-harm) and shows the Tele-MANAS helpline (14416, India), regardless of model output.

## 📈 Evaluation Metrics

- **F1 micro / macro:** overall and per-emotion balance of precision and recall.
- **Precision:** how many flagged emotions were correct.
- **Recall:** how many real emotions were found. This matters most, so concerning signals are not missed.

Results are saved to `results/metrics.json` after training. See `RESEARCH_PAPER.md` for the results table.

## 📁 Project Structure

```
EchoBuddy/
│
├── index.html          # Chat + Parent Dashboard + Wisdom Library
├── server.py           # FastAPI backend (/analyze)
├── train.py            # DistilBERT fine-tuning on GoEmotions
├── trend_lstm.py       # LSTM mood-trend proof of concept
├── emotion_map.py      # Emotion grouping and safety rule
├── requirements.txt
├── RESEARCH_PAPER.md
├── LICENSE
├── .gitignore
├── model/              # trained model is saved here (not uploaded)
└── results/            # metrics.json
```

## ▶️ How to Run

1. Clone this repository and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Train the emotion model (use a GPU such as free Google Colab; about 15-20 minutes):
   ```
   python train.py
   ```
3. (Optional) Run the mood-trend proof of concept:
   ```
   python trend_lstm.py
   ```
4. Start the app:
   ```
   uvicorn server:app --reload
   ```
   Open http://127.0.0.1:8000 in your browser.

Quick demo without training: just open `index.html` in a browser (keyword mode).

## 🔒 Ethics and Privacy

- Parents see only a summary, not the full chat.
- Clear consent from the student and the guardian is required before use.
- The system gives signals, not diagnoses. Both false alarms and missed cases are possible.
- Religious and moral content is presented respectfully and without pressure.
- Serious statements always show a helpline and alert the guardian.

## 🚀 Future Work

- Fine-tune on India-specific and Hinglish student conversations.
- Semantic retrieval for the wisdom library using sentence embeddings.
- Validate the trend model with expert-labelled, ethically collected data.
- Multilingual support (Hindi and regional languages).

## 👨‍💻 Project

**EchoBuddy: An Early Emotional Wellbeing Companion for Students**

Created as an Artificial Intelligence (Machine Learning / Deep Learning) project.
Author: `<Your Name>` | Contact: `<Your Email>`
