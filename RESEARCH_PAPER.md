# EchoBuddy: An Early Emotional Wellbeing Companion for Students Using Deep Learning and Value-Based Guidance

**Author:** <Krishna Sharma>, <Btech CSE>, <COER University>, <Roorkee,India>
**Email:** <www.krishnasharma21@gmail.com>

> Fill every `<...>` placeholder and the Results table (Section 6) with your own numbers after running the code. Do not publish numbers you have not produced yourself.

---

## Abstract

Students today face growing academic pressure, social comparison and loneliness, and early signs of stress often go unnoticed by parents and teachers. This paper presents EchoBuddy, an AI companion that chats with students as a friend, detects emotions in their messages using a fine-tuned DistilBERT model, tracks mood trends over time, and shares a privacy-respecting summary with parents through a dashboard. To encourage optimism and moral values, EchoBuddy retrieves verses from the Bhagavad Gita, the Dhammapada and stories from the Ramayana, along with real-life stories of perseverance, and explains them in simple language. The emotion classifier is trained on the GoEmotions dataset (28 emotion labels), and a safety layer always raises an alert for serious statements. A small LSTM is explored as a proof of concept for trend detection. EchoBuddy is designed as an early-signal support tool and not as a diagnostic system. We report the model's performance as <F1 macro = ___, F1 micro = ___> and discuss ethical considerations and limitations.

**Keywords:** emotion detection, student wellbeing, DistilBERT, LSTM, conversational AI, parent dashboard, value education

---

## 1. Introduction

Mental health concerns among students, including stress, anxiety and low mood, are rising, and many young people hesitate to talk openly about their feelings. Parents and teachers often notice problems only when they become serious. A friendly, always-available conversational companion could help students express themselves, and could offer early signals to the adults who care for them.

At the same time, many young people are losing touch with moral and ethical values that earlier generations learned through stories and scriptures. Teachings such as the Bhagavad Gita's message on effort without fear of results can offer comfort and perspective when explained in a simple way.

This work proposes EchoBuddy, which combines (i) emotion detection from text using deep learning, (ii) mood-trend monitoring, (iii) a parent dashboard with summaries, and (iv) a library of motivational and value-based content.

## 2. Related Work

Transformer models such as BERT [2] have become the standard for text classification, and DistilBERT [3] provides a lighter and faster alternative with comparable accuracy. GoEmotions [1] is a large annotated dataset of Reddit comments with 27 emotion categories plus neutral. Stress detection from social media text has been studied with datasets such as Dreaddit [5]. Recurrent networks such as LSTMs [4] are widely used for sequence data, which makes them suitable for modelling mood over time. Conversational agents for mental wellbeing exist, but few combine emotion tracking, parent summaries and culturally rooted value guidance for students. <Add 3-5 more papers you read, with citations.>

## 3. Problem Statement and Objectives

**Problem:** Early emotional distress in students often goes undetected, and young people lack a safe, friendly space to share daily problems and receive positive guidance.

**Objectives:**
1. Build a friendly chat assistant for students.
2. Detect emotions in messages with a fine-tuned transformer.
3. Identify concerning mood trends over time.
4. Provide parents with a concise summary and not the full conversation.
5. Deliver motivation and moral guidance through verses and real-life stories in simple language.

## 4. Methodology

### 4.1 System Architecture
EchoBuddy has five modules: (1) chat interface, (2) emotion detection (DistilBERT), (3) risk and trend scoring, (4) parent dashboard, (5) wisdom library. A safety layer monitors every message. <Insert an architecture diagram here.>

### 4.2 Dataset
We use GoEmotions (about 58,000 comments, 27 emotions plus neutral) with its standard train/validation/test splits. The 28 labels are grouped into four wellbeing categories: sad (sadness, grief, disappointment, remorse), anxiety (fear, nervousness), anger (anger, annoyance, disapproval, disgust) and happy (joy, amusement, excitement, gratitude, love, optimism, pride, admiration, relief). Categories not covered by the dataset, such as loneliness, failure and social comparison, are currently detected by keywords.

### 4.3 Emotion Classifier
We fine-tune `distilbert-base-uncased` for multi-label classification with a sigmoid output layer and binary cross-entropy loss. Hyperparameters: max sequence length 64, batch size 32, learning rate 5e-5, 3 epochs, decision threshold 0.3, seed 42. The best checkpoint is chosen by macro F1 on the validation set.

### 4.4 Mood-Trend Model
Each message produces a vector of group scores [sad, anxiety, anger, happy]. An LSTM reads the last 10 vectors and outputs a probability that the trend is concerning. In this version, the LSTM is trained on synthetic sequences as a proof of concept only; the deployed dashboard also uses a transparent rule based on the share of negative messages (moderate at 50% or more, high at 75% or more over the recent window).

### 4.5 Safety Layer
A rule-based filter detects serious statements (for example about self-harm). It always triggers a supportive response with a helpline (Tele-MANAS 14416, India) and an urgent alert to the guardian, independent of the model's output.

### 4.6 Wisdom Library
The library contains 31 items: ten Bhagavad Gita verses in Sanskrit with English explanations, three Dhammapada verses, five Ramayana stories, nine real-life stories (for example APJ Abdul Kalam, Srinivasa Ramanujan, Helen Keller) and four calming or gratitude exercises. Each item is tagged with an emotional theme. The chatbot selects an item matching the detected emotion. In the current version retrieval is tag-based; semantic retrieval using sentence embeddings is planned.

### 4.7 Parent Dashboard
The dashboard shows an overall concern level (low, moderate, high), a plain-language summary, emotion counts and a mood-trend chart. Parents receive only summaries; the chat text is not shown.

## 5. Implementation

The system is implemented in Python (PyTorch, Hugging Face Transformers and Datasets) with a FastAPI backend exposing an `/analyze` endpoint. The front end uses HTML, CSS and JavaScript. If the model server is unavailable, the front end falls back to keyword-based detection. Training was run on <GPU/Colab details> and took about <time>. Code: <GitHub link>.

## 6. Results and Discussion

Replace the blanks with the values from `results/metrics.json` after running `python train.py`.

| Metric (GoEmotions test set) | Value |
|---|---|
| F1 micro | <___> |
| F1 macro | <___> |
| Precision macro | <___> |
| Recall macro | <___> |

**Discussion (write after seeing your results):** Compare your scores with the GoEmotions baseline reported in [1]. Mention which emotions were detected well and which were confused (for example, similar emotions such as annoyance and anger). You may add a confusion analysis, example predictions and a screenshot of the dashboard.

**Trend model:** The LSTM reached <___> accuracy on synthetic data. This number only shows that the model can learn the pattern we defined; it is not evidence of clinical accuracy.

## 7. Ethical Considerations

- **Not a diagnosis:** EchoBuddy gives early signals only; professional assessment is essential.
- **Consent and privacy:** Both student and guardian must consent; data should be encrypted and minimal; parents see summaries only.
- **Children's data:** Follow applicable data-protection laws (for example India's Digital Personal Data Protection Act, 2023, which has special provisions for children's data).
- **Errors:** False alarms may cause worry, and missed cases may give false reassurance.
- **Cultural and religious content:** Presented respectfully, without pressure, and from reliable translations; the system can include texts from other traditions.
- **Over-reliance:** EchoBuddy should encourage real conversations with family and friends.

## 8. Limitations

1. GoEmotions is built from Reddit comments, not student conversations, so domain shift is likely.
2. Loneliness, failure and comparison are detected by keywords, not learned.
3. The trend model has not been validated on real, expert-labelled data.
4. The system is English-only; Hindi and Hinglish are not yet supported.
5. Sarcasm and indirect expression may be missed.

## 9. Conclusion and Future Work

EchoBuddy shows how an emotion-aware chat companion with a parent summary and value-based guidance can support student wellbeing. Future work includes fine-tuning on student-specific and Hinglish data, semantic retrieval of wisdom content, validation with counsellors, a mobile app, and a pilot study with consent in a school setting.

## References

[1] D. Demszky, D. Movshovitz-Attias, J. Ko, A. Cowen, G. Nemade, and S. Ravi, "GoEmotions: A Dataset of Fine-Grained Emotions," in *Proc. 58th Annual Meeting of the ACL*, 2020.

[2] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. NAACL-HLT*, 2019.

[3] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," arXiv:1910.01108, 2019.

[4] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735-1780, 1997.

[5] E. Turcan and K. McKeown, "Dreaddit: A Reddit Dataset for Stress Analysis in Social Media," in *Proc. 10th International Workshop on Health Text Mining and Information Analysis (LOUHI)*, 2019.

[6] T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," in *Proc. EMNLP: System Demonstrations*, 2020.
