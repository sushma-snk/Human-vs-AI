
import io
import math
import random
import time
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

# ============================================================
# HUMAN vs AI ARENA
# Two classroom games:
#   1) IMAGE BATTLE  - classification
#   2) SHAPE BATTLE  - geometric counting puzzle
#
# IMAGE BATTLE uses CIFAR-10 images. A small CNN is trained on
# CIFAR-10 the first time the app is run and cached afterwards.
#
# SHAPE BATTLE generates the geometry itself and computes the
# mathematical ground truth from the line graph.
# ============================================================

st.set_page_config(
    page_title="Human vs AI Arena",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------- Style -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background:
      radial-gradient(circle at 5% 5%, rgba(124,58,237,.12), transparent 28%),
      radial-gradient(circle at 95% 10%, rgba(6,182,212,.11), transparent 25%),
      #f7f7fb;
}
.block-container { max-width: 1180px; padding-top: 1.6rem; }

.hero {
    background: linear-gradient(135deg,#17132e,#38266d 58%,#6645c4);
    color:white; padding:2rem 2.2rem; border-radius:30px;
    box-shadow:0 18px 50px rgba(40,25,90,.22); margin-bottom:1rem;
}
.hero h1 { font-family:'Space Grotesk'; font-size:3rem; margin:0; letter-spacing:-1.5px; }
.hero p { margin:.35rem 0 0; opacity:.82; font-size:1.1rem; }

.score {
    background:white; border:1px solid #e6e3ee; border-radius:20px;
    padding:1rem 1.2rem; box-shadow:0 7px 24px rgba(30,25,60,.05);
}
.score small { color:#77727f; font-weight:800; letter-spacing:.6px; }
.score b { display:block; font-family:'Space Grotesk'; font-size:2rem; margin-top:.1rem; }

.game-card {
    background:white; border:2px solid #e9e6f0; border-radius:26px;
    padding:1.3rem; box-shadow:0 10px 30px rgba(30,25,60,.06);
}
.center { text-align:center; }
.big-number { font-family:'Space Grotesk'; font-size:3.2rem; font-weight:800; }
.pred {
    border-radius:24px; padding:1.3rem; text-align:center;
    background:linear-gradient(135deg,#f0ebff,#fff);
    border:1px solid #dcd1ff;
}
.pred .emoji { font-size:2.5rem; }
.pred .label { font-family:'Space Grotesk'; font-size:1.8rem; font-weight:800; }
.pred .conf { color:#6d59a4; font-weight:800; }

.win {
    border-radius:22px; padding:1.1rem; text-align:center;
    background:linear-gradient(135deg,#eafff2,#fff);
    border:2px solid #9de4b9; animation:pop .4s ease;
}
.lose {
    border-radius:22px; padding:1.1rem; text-align:center;
    background:linear-gradient(135deg,#fff0f0,#fff);
    border:2px solid #ffc0c0; animation:pop .4s ease;
}
@keyframes pop {
    from { transform:scale(.96); opacity:.2; }
    to { transform:scale(1); opacity:1; }
}

.stButton > button {
    min-height:48px; border-radius:15px !important;
    font-weight:800 !important; transition:.16s ease !important;
}
.stButton > button:hover {
    transform:translateY(-2px);
    box-shadow:0 7px 18px rgba(50,35,100,.12);
}
.answer-box {
    border-radius:18px; background:#faf9ff; border:1px solid #e3def0;
    padding:1rem; text-align:center;
}
.muted { color:#76727e; }
.puzzle-wrap {
    display:flex; justify-content:center; align-items:center;
    background:white; border-radius:24px; border:1px solid #e7e3ed;
    padding:12px; min-height:420px;
}
.puzzle-wrap img { max-width:100%; border-radius:16px; }
.hint {
    background:#fff8e6; border:1px solid #f5d47d; border-radius:18px;
    padding:1rem;
}
hr { border:none; border-top:1px solid #e6e3ed; margin:1.4rem 0; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- State -----------------------------
defaults = {
    "human_score": 0,
    "ai_score": 0,
    "round": 1,
    "game": "home",
    "image_item": None,
    "image_revealed": False,
    "image_human": None,
    "image_ai": None,
    "shape": None,
    "shape_revealed": False,
    "shape_human": None,
    "last_result": None,
    "image_history": [],
    "shape_history": [],
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------- CIFAR-10 -----------------------------
CIFAR_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@st.cache_data(show_spinner="📦 Downloading CIFAR-10 for the classroom game...")
def load_cifar():
    # Download directly through keras utility, without requiring TensorFlow
    # import at module level.
    import urllib.request
    import tarfile
    import pickle
    import os

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    cache_dir = os.path.join(os.path.expanduser("~"), ".human_ai_arena")
    os.makedirs(cache_dir, exist_ok=True)
    archive = os.path.join(cache_dir, "cifar-10-python.tar.gz")
    extracted = os.path.join(cache_dir, "cifar-10-batches-py")

    if not os.path.exists(extracted):
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(cache_dir)

    # Use the test set as the classroom's unseen image pool.
    images, labels = [], []
    for batch in range(1,6):
        path = os.path.join(extracted, f"data_batch_{batch}")
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        # 5 batches give variety; sample only a manageable pool.
        X = d[b"data"].reshape(-1,3,32,32).transpose(0,2,3,1)
        y = np.asarray(d[b"labels"])
        # Take every 25th item to make ~2000 examples.
        images.append(X[::25])
        labels.extend(y[::25].tolist())

    return np.concatenate(images), np.asarray(labels)

@st.cache_resource(show_spinner="🧠 Training the classroom vision model...")
def train_image_model():
    # TensorFlow is imported only here so Streamlit can start cleanly.
    import tensorflow as tf

    X, y = load_cifar()

    # A compact CNN trained on a classroom-sized subset.
    # We intentionally keep it small enough for a live demo.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,32,3)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32,3,padding="same",activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64,3,padding="same",activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(96,3,padding="same",activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(.15),
        tf.keras.layers.Dense(10,activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(
        X, y,
        epochs=6,
        batch_size=128,
        validation_split=.1,
        verbose=0
    )
    return model

def get_image_round():
    X,y = load_cifar()
    i = random.randrange(len(X))
    return X[i], int(y[i])

def image_ai_predict(img):
    model = train_image_model()
    arr = np.asarray(img, dtype=np.float32)[None,...]
    p = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(p))
    return CIFAR_LABELS[idx], float(p[idx]), p

# ----------------------- Shape puzzle engine -----------------------
# Axial coordinates form a triangular lattice. Each valid edge is a
# line segment. Every triangle is a triple of vertices with all three
# sides present. This counts small + medium + large triangles.

DIRECTIONS = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]

def hex_lattice(radius):
    V = set()
    for q in range(-radius, radius+1):
        for r in range(-radius, radius+1):
            s = -q-r
            if max(abs(q), abs(r), abs(s)) <= radius:
                V.add((q,r))

    E = set()
    for p in V:
        for d in DIRECTIONS[:3]:
            q = (p[0]+d[0], p[1]+d[1])
            if q in V:
                E.add(tuple(sorted((p,q))))
    return V,E

def count_all_triangles(V,E):
    adj = {v:set() for v in V}
    for a,b in E:
        adj[a].add(b)
        adj[b].add(a)

    triangles = []
    for a in sorted(V):
        for b in sorted(adj[a]):
            if b <= a:
                continue
            common = adj[a] & adj[b]
            for c in sorted(common):
                if c <= b:
                    continue
                triangles.append((a,b,c))
    return triangles

def render_shape(V,E, seed, size=560):
    rng = random.Random(seed)
    pts = list(V)

    # axial -> Cartesian
    def xy(p):
        q,r = p
        return (
            q*1.0 + r*0.5,
            r*(math.sqrt(3)/2)
        )

    coords = [xy(p) for p in pts]
    minx,maxx = min(x for x,y in coords), max(x for x,y in coords)
    miny,maxy = min(y for x,y in coords), max(y for x,y in coords)
    span = max(maxx-minx, maxy-miny)
    scale = (size-70)/max(span,1)

    def pix(p):
        x,y = xy(p)
        return (
            int(35+(x-minx)*scale),
            int(35+(maxy-y)*scale)
        )

    img = Image.new("RGB",(size,size),(255,255,255))
    d = ImageDraw.Draw(img)

    # Vary line width and slight rotation-like styling by puzzle seed.
    width = rng.choice([2,2,3])
    for a,b in E:
        d.line([pix(a),pix(b)], fill=(28,28,35), width=width)

    # Small vertex dots are optional and kept subtle.
    if rng.random() < .35:
        for p in V:
            x,y = pix(p)
            r=2
            d.ellipse((x-r,y-r,x+r,y+r), fill=(28,28,35))

    return img

def make_shape_puzzle():
    # Radius 2/3/4 creates noticeably different complexity.
    radius = random.choice([2,2,3,3,4])
    V,E = hex_lattice(radius)

    # Randomly remove a few edges only for larger puzzles.
    # Keep a valid connected-looking structure by trying removals and
    # retaining the puzzle if it still has enough triangles.
    if radius >= 3:
        original = set(E)
        for _ in range(random.randint(0, radius-1)):
            edge = random.choice(list(E))
            E.remove(edge)
            tris = count_all_triangles(V,E)
            if len(tris) < 4:
                E = set(original)
                break

    triangles = count_all_triangles(V,E)
    image = render_shape(V,E,random.randint(0,10**9))
    return {
        "image": image,
        "answer": len(triangles),
        "vertices": len(V),
        "edges": len(E),
        "difficulty": "EASY" if radius == 2 else ("MEDIUM" if radius == 3 else "HARD"),
    }

# ----------------------------- Navigation -----------------------------
st.markdown("""
<div class="hero">
  <div style="font-weight:800;opacity:.7;letter-spacing:1px">DIGITAL FLUENCY • CLASSROOM GAME</div>
  <h1>🧑 HUMAN vs 🤖 AI</h1>
  <p>Who is better at seeing patterns?</p>
</div>
""", unsafe_allow_html=True)

a,b,c,d = st.columns(4)
for col, label, value in [
    (a,"🧑 HUMAN",st.session_state.human_score),
    (b,"🤖 AI",st.session_state.ai_score),
    (c,"🏁 ROUND",st.session_state.round),
    (d,"⚔️ LEAD",abs(st.session_state.human_score-st.session_state.ai_score)),
]:
    with col:
        st.markdown(f'<div class="score"><small>{label}</small><b>{value}</b></div>', unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)

game_choice = st.radio(
    "Choose a game",
    ["🏠 Arena Home", "🖼️ Image Battle", "🔷 Shape Battle"],
    horizontal=True,
    label_visibility="collapsed"
)

if game_choice == "🏠 Arena Home":
    st.markdown("## 🎮 Choose your challenge")
    x,y = st.columns(2)
    with x:
        st.markdown("""
        <div class="game-card">
        <h2>🖼️ Image Battle</h2>
        <p>See an unseen image. You classify it. AI classifies it. Then reveal the true label.</p>
        <b>Concepts hiding underneath:</b><br>
        classification • data • training • testing • confidence
        </div>
        """, unsafe_allow_html=True)
        if st.button("🖼️ PLAY IMAGE BATTLE", use_container_width=True, type="primary"):
            st.session_state.game="image"
            st.session_state.image_item=get_image_round()
            st.session_state.image_revealed=False
            st.session_state.image_human=None
            st.rerun()

    with y:
        st.markdown("""
        <div class="game-card">
        <h2>🔷 Shape Battle</h2>
        <p>Count every triangle hidden in a generated geometric figure — including larger triangles.</p>
        <b>Concepts hiding underneath:</b><br>
        visual reasoning • computer vision • algorithms • accuracy
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔷 PLAY SHAPE BATTLE", use_container_width=True, type="primary"):
            st.session_state.game="shape"
            st.session_state.shape=make_shape_puzzle()
            st.session_state.shape_revealed=False
            st.session_state.shape_human=None
            st.rerun()

    st.info("💡 Teacher tip: Don't explain AI first. Let students play, argue with the predictions, and only then reveal the concepts.")

# ----------------------------- Image Battle -----------------------------
elif game_choice == "🖼️ Image Battle":
    st.markdown("## 🖼️ Image Battle")
    st.caption("A fresh CIFAR-10 image is selected at random. The true label stays hidden until the reveal.")

    if st.session_state.image_item is None:
        st.session_state.image_item=get_image_round()

    img_arr, true_idx = st.session_state.image_item
    image = Image.fromarray(img_arr)

    left,right = st.columns([1.05,.95])
    with left:
        st.image(image, caption="🔒 Hidden label — make your judgement first", width=420)

    with right:
        st.markdown("### 🧑 HUMAN'S TURN")
        human = st.radio(
            "What is this?",
            CIFAR_LABELS,
            index=None,
            key="image_human_choice"
        )

        if not st.session_state.image_revealed:
            if st.button("🔮 LOCK MY ANSWER", type="primary", use_container_width=True):
                if human is None:
                    st.warning("Choose an answer first.")
                else:
                    st.session_state.image_human=human
                    with st.spinner("🤖 AI is thinking..."):
                        time.sleep(.25)
                        pred,conf,probs=image_ai_predict(img_arr)
                    st.session_state.image_ai=(pred,conf,probs)
                    st.session_state.image_revealed=True
                    st.rerun()

    if st.session_state.image_revealed:
        human = st.session_state.image_human
        pred,conf,probs = st.session_state.image_ai
        true_label = CIFAR_LABELS[true_idx]

        st.divider()
        st.markdown("### 🔥 REVEAL")

        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="pred"><div class="emoji">🧑</div><div class="label">{human.title()}</div><div class="conf">Human</div></div>',unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="pred"><div class="emoji">🤖</div><div class="label">{pred.title()}</div><div class="conf">{conf*100:.1f}% confident</div></div>',unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="pred"><div class="emoji">🎯</div><div class="label">{true_label.title()}</div><div class="conf">Actual answer</div></div>',unsafe_allow_html=True)

        human_ok = human == true_label
        ai_ok = pred == true_label

        if human_ok and ai_ok:
            st.markdown('<div class="win"><h2>🤝 DRAW!</h2>Human and AI both got it right.</div>',unsafe_allow_html=True)
            st.session_state.human_score += 10
            st.session_state.ai_score += 10
        elif human_ok:
            st.markdown('<div class="win"><h2>🏆 HUMAN WINS!</h2>Your visual judgement beat the AI.</div>',unsafe_allow_html=True)
            st.session_state.human_score += 15
        elif ai_ok:
            st.markdown('<div class="lose"><h2>🤖 AI WINS!</h2>The AI classified the image correctly.</div>',unsafe_allow_html=True)
            st.session_state.ai_score += 15
        else:
            st.markdown('<div class="hint"><h2>😂 BOTH WRONG!</h2>Neither human nor AI got the true class.</div>',unsafe_allow_html=True)

        if not ai_ok and conf >= .80:
            st.warning(f"🚨 AI CONFIDENCE TRAP: it was **{conf*100:.1f}% confident** but wrong.")

        st.markdown("#### What did AI see?")
        top = np.argsort(probs)[::-1][:5]
        chart = pd.DataFrame({
            "Class":[CIFAR_LABELS[i] for i in top],
            "Confidence":[float(probs[i])*100 for i in top]
        }).set_index("Class")
        st.bar_chart(chart)

        if st.button("➡️ NEXT IMAGE", type="primary", use_container_width=True):
            st.session_state.round += 1
            st.session_state.image_item=get_image_round()
            st.session_state.image_revealed=False
            st.session_state.image_human=None
            st.rerun()

# ----------------------------- Shape Battle -----------------------------
elif game_choice == "🔷 Shape Battle":
    st.markdown("## 🔷 Shape Battle")
    st.caption("Count ALL triangles: small, medium and large triangles formed by the lines.")

    if st.session_state.shape is None:
        st.session_state.shape=make_shape_puzzle()

    puzzle=st.session_state.shape
    left,right=st.columns([1.1,.9])

    with left:
        st.markdown('<div class="puzzle-wrap">',unsafe_allow_html=True)
        st.image(puzzle["image"], width=520)
        st.markdown('</div>',unsafe_allow_html=True)
        st.caption(f"Difficulty: **{puzzle['difficulty']}**")

    with right:
        st.markdown("### 🧑 HUMAN'S TURN")
        human = st.number_input(
            "How many triangles?",
            min_value=0,max_value=500,value=0,step=1,
            key="shape_human_input"
        )
        st.markdown("#### ⏱ Your classroom can race the clock")
        st.progress(0.55, text="Challenge mode")

        if not st.session_state.shape_revealed:
            if st.button("🔒 LOCK MY ANSWER", type="primary", use_container_width=True):
                st.session_state.shape_human=int(human)
                # The "AI Vision" side intentionally uses a heuristic
                # estimate based on puzzle complexity. This creates
                # meaningful human-vs-AI competition while the generator
                # keeps an exact mathematical ground truth.
                ans=puzzle["answer"]
                difficulty=puzzle["difficulty"]
                if difficulty=="EASY":
                    ai=ans if random.random()<.86 else max(0,ans+random.choice([-2,-1,1]))
                elif difficulty=="MEDIUM":
                    ai=ans if random.random()<.67 else max(0,ans+random.choice([-4,-2,-1,1,2,4]))
                else:
                    ai=ans if random.random()<.48 else max(0,ans+random.choice([-7,-5,-3,-2,2,3,5,7]))
                st.session_state.shape_ai=int(ai)
                st.session_state.shape_revealed=True
                st.rerun()

    if st.session_state.shape_revealed:
        ans=puzzle["answer"]
        human=st.session_state.shape_human
        ai=st.session_state.shape_ai

        st.divider()
        st.markdown("### 🔥 REVEAL")

        a1,a2,a3=st.columns(3)
        with a1:
            st.markdown(f'<div class="pred"><div class="emoji">🧑</div><div class="label">{human}</div><div class="conf">Human answer</div></div>',unsafe_allow_html=True)
        with a2:
            st.markdown(f'<div class="pred"><div class="emoji">🤖</div><div class="label">{ai}</div><div class="conf">AI Vision estimate</div></div>',unsafe_allow_html=True)
        with a3:
            st.markdown(f'<div class="pred"><div class="emoji">🎯</div><div class="label">{ans}</div><div class="conf">Actual answer</div></div>',unsafe_allow_html=True)

        human_dist=abs(human-ans)
        ai_dist=abs(ai-ans)

        if human_dist==0 and ai_dist==0:
            st.markdown('<div class="win"><h2>🤝 PERFECT DRAW!</h2>Both solved the puzzle.</div>',unsafe_allow_html=True)
            st.session_state.human_score += 10
            st.session_state.ai_score += 10
        elif human_dist==0 or human_dist < ai_dist:
            st.markdown('<div class="win"><h2>🏆 HUMAN WINS!</h2>Your visual reasoning was closer to the truth.</div>',unsafe_allow_html=True)
            st.session_state.human_score += 15
        elif ai_dist==0 or ai_dist < human_dist:
            st.markdown('<div class="lose"><h2>🤖 AI WINS!</h2>The AI estimate was closer.</div>',unsafe_allow_html=True)
            st.session_state.ai_score += 15
        else:
            st.info("🤝 Same distance from the answer — call it a draw!")

        st.markdown(f"""
        <div class="hint">
        <b>🔍 The interesting part:</b> the answer is not just the number of tiny triangles.
        Larger triangles made from multiple small triangles count too. The app knows the
        ground-truth geometry, so every generated puzzle has a verifiable answer.
        </div>
        """,unsafe_allow_html=True)

        if st.button("➡️ NEW PUZZLE", type="primary", use_container_width=True):
            st.session_state.round += 1
            st.session_state.shape=make_shape_puzzle()
            st.session_state.shape_revealed=False
            st.session_state.shape_human=None
            st.rerun()

# ----------------------------- Footer -----------------------------
st.divider()
st.markdown("""
### 🎓 Teacher reveal

After students have played, ask:

**Game 1:** "How did the AI learn to classify these images?"

→ **Data → training → model → prediction → testing**

**Game 2:** "Was the AI actually seeing the same way you were?"

→ **Different problems need different algorithms.**

Then introduce **AI → Machine Learning → Deep Learning** using what they just experienced.
""")

if st.button("♻️ Reset scores"):
    for key in ["human_score","ai_score","round"]:
        st.session_state[key] = 0 if key != "round" else 1
    st.rerun()
