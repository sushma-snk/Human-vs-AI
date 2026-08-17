# 🧑 HUMAN vs 🤖 AI — Classroom Game Arena

A Streamlit classroom application for a first Digital Fluency session with B.Des Product Design and Interaction Design students.

The app contains two games under one Human-vs-AI scoreboard.

## Game 1 — 🖼️ Image Battle

A random CIFAR-10 test image is shown. Students classify it without knowing the true label. The trained CNN predicts independently. The true label is revealed and the scores are updated.

The flow is:

```text
Random unseen image
       ↓
Human classification
       ↓
AI classification
       ↓
Reveal true label
       ↓
Human vs AI score
```

The game demonstrates classification, training data, testing data, confidence and errors.

## Game 2 — 🔷 Shape Battle

A new geometric puzzle is generated programmatically every round.

The puzzle is based on a triangular lattice. The program stores the vertices and edges, then enumerates all valid triangles, including larger triangles formed from smaller ones.

The students see only the rendered puzzle and enter their count.

The app compares:

- Human answer
- AI Vision estimate
- Exact mathematical ground truth

### Important honesty note

The current Shape Battle's "AI Vision estimate" is a **difficulty-aware simulated AI opponent**, not a neural computer-vision model. This is intentional for the first classroom version so the game can reliably produce both human and AI wins while preserving an exact ground truth.

If you want a technically stronger second version, the AI side can be replaced with an actual OpenCV-based vision pipeline that reconstructs the line graph from the rendered image.

## Image Battle model

The app downloads CIFAR-10 and trains a compact CNN on a small, balanced classroom subset. The model is cached by Streamlit after the first successful run.

The first startup can therefore take several minutes.

## Deployment

Use **Python 3.12**.

Streamlit Cloud:
1. Push the files to GitHub.
2. Create/deploy the Streamlit app.
3. Select `app.py`.
4. In Advanced settings select Python 3.12.
5. Deploy.

The earlier TensorFlow/Python 3.14 dependency problem is avoided by using Python 3.12.

## Local

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Classroom use

Do not explain AI before the game.

A suggested sequence:

1. Put the app on the projector.
2. Split the class into "Human" and "AI" mentally / by team.
3. Let the room discuss the human answer.
4. Reveal AI.
5. Reveal the true answer.
6. Keep a visible score.
7. Move to the Shape Battle.
8. Only after playing, ask what the students think "AI", "ML", "DL", "training", "testing" and "classification" mean.

The application is designed to turn the concepts into something students experience rather than definitions they memorise.
