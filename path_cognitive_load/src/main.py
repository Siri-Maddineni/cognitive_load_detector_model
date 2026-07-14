"""Main GUI — Cognitive Load Detection System (FINAL)"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import cv2
import threading
import time
import math
import random
from datetime import datetime
from PIL import Image, ImageTk

try:
    from src.mouse_tracker import MouseTracker
except ImportError:
    MouseTracker = None
try:
    from src.database import Database
except ImportError:
    Database = None
try:
    from src.analyzer import ADHDDyslexiaAnalyzer
except ImportError:
    ADHDDyslexiaAnalyzer = None
try:
    from src.models import ModelPredictor
except ImportError:
    ModelPredictor = None

THEMES = {
    'light': dict(bg='#f0f0f0', fg='#333333', card_bg='#ffffff',
                  card_fg='#333333', header_bg='#2C3E50',
                  header_fg='#ffffff', button_bg='#4CAF50',
                  button_fg='#ffffff', accent='#2196F3'),
    'dark': dict(bg='#1e1e1e', fg='#ffffff', card_bg='#2d2d2d',
                 card_fg='#ffffff', header_bg='#1a1a2e',
                 header_fg='#ffffff', button_bg='#0f3460',
                 button_fg='#ffffff', accent='#e94560'),
}

QUESTIONS = [
    dict(text="What is 12 + 19?", answer="31", type="arithmetic", difficulty=0),
    dict(text="Complete the pattern: 2, 4, 6, 8, __?", answer="10", type="pattern", difficulty=0),
    dict(text="What is the opposite of 'hot'?", answer="cold", type="verbal", difficulty=0),
    dict(text="How many days are in a week?", answer="7", type="knowledge", difficulty=0),
    dict(text="What is 15% of 200?", answer="30", type="arithmetic", difficulty=1),
    dict(text="What comes next: 3, 6, 12, 24, __?", answer="48", type="pattern", difficulty=1),
    dict(text="If 5 notebooks cost $45,\nhow much do 3 cost?", answer="27", type="reasoning", difficulty=1),
    dict(text="What is the capital of France?", answer="paris", type="knowledge", difficulty=1),
    dict(text="What is 17 × 23?", answer="391", type="arithmetic", difficulty=2),
    dict(text="Next in sequence:\n1, 1, 2, 3, 5, 8, __?", answer="13", type="pattern", difficulty=2),
    dict(text="A car travels 240 km in 3 hours.\nAverage speed in km/h?", answer="80", type="reasoning", difficulty=2),
    dict(text="What is the square root of 144?", answer="12", type="arithmetic", difficulty=2),
]

LOAD_LABELS = ['Low', 'Medium', 'High']
LOAD_COLORS = {'Low': '#4CAF50', 'Medium': '#FFC107', 'High': '#F44336'}


class CognitiveLoadApp:

    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Cognitive Load Detection — ADHD & Dyslexia")
        self.root.geometry("1600x950")
        self.root.minsize(1200, 700)

        self.theme = 'light'
        self.is_running = True
        self.session_active = False
        self.current_student = None
        self.session_start = None

        self.cap = cv2.VideoCapture(0)
        self.cam_ok = self.cap.isOpened()
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')

        self.face_ok = False
        self.face_frames = 0
        self.total_frames = 0
        self.blink_count = 0
        self._eyes_open = True
        self._open_streak = 0
        self._last_blink_t = 0.0
        self.eye_str = "Waiting…"
        self.head_offset = (0.0, 0.0)
        self.track_start = time.time()

        self.mouse_tracker = MouseTracker() if MouseTracker else None
        self.cur_speed = 0.0
        self.cur_jitter = 0.0
        self.cur_idle = 0.0

        self.questions_list = []
        self.q_idx = 0
        self.q_start = 0.0
        self.responses = []
        self.load_history = []

        self.video_label = None
        self.lbl_face = None
        self.lbl_blink = None
        self.lbl_eye = None
        self.lbl_head = None
        self.lbl_cursor = None
        self.lbl_idle = None

        self.db = Database() if Database else None
        self.model_predictor = ModelPredictor() if ModelPredictor else None
        self.submitting_answer = False
        self.camera_processing = True

        threading.Thread(target=self._cam_loop, daemon=True).start()
        self.root.bind('<Motion>', self._on_mouse)
        self._build_login()
        self._tick()

    def _cam_loop(self):
        BLINK_CD = 0.3
        while self.is_running:
            if not self.camera_processing:
                time.sleep(0.1)
                continue
            if not self.cam_ok:
                time.sleep(0.5)
                continue
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            self.total_frames += 1

            if len(faces) > 0:
                self.face_ok = True
                self.face_frames += 1
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi, 1.1, 5)
                ne = min(len(eyes), 2)
                for (ex, ey, ew, eh) in eyes[:2]:
                    cv2.rectangle(frame, (x+ex, y+ey),
                                  (x+ex+ew, y+ey+eh), (255, 0, 0), 1)
                if ne >= 2:
                    self._open_streak += 1
                    if self._open_streak >= 2:
                        self._eyes_open = True
                    self.eye_str = "Both eyes open"
                else:
                    self._open_streak = 0
                    if self._eyes_open:
                        now = time.time()
                        if now - self._last_blink_t > BLINK_CD:
                            self.blink_count += 1
                            self._last_blink_t = now
                        self._eyes_open = False
                    self.eye_str = "One eye" if ne == 1 else "Eyes closed/hidden"
                fh, fw = frame.shape[:2]
                self.head_offset = ((x + w/2 - fw/2) / (fw/2),
                                    (y + h/2 - fh/2) / (fh/2))
            else:
                self.face_ok = False
                self.eye_str = "No face detected"
                self._open_streak = 0

            tag = "QUIZ" if self.session_active else "READY"
            cv2.putText(frame, tag, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            col = (0, 255, 0) if self.face_ok else (0, 0, 255)
            cv2.putText(frame, "Face: YES" if self.face_ok else "Face: NO",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            el = max(time.time() - self.track_start, 1)
            bpm = self.blink_count / el * 60
            cv2.putText(frame, f"Blinks: {self.blink_count} ({bpm:.0f}/min)",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            with self.frame_lock:
                self.current_frame = frame.copy()
            time.sleep(0.033)

    def _on_mouse(self, ev):
        if self.mouse_tracker:
            self.mouse_tracker.update(ev.x_root, ev.y_root)
            self.cur_speed = self.mouse_tracker.avg_speed
            self.cur_jitter = self.mouse_tracker.jitter
            self.cur_idle = self.mouse_tracker.idle_seconds

    def _tick(self):
        if not self.is_running:
            return
        if self.mouse_tracker:
            self.cur_idle = self.mouse_tracker.idle_seconds
        with self.frame_lock:
            frm = self.current_frame
        if frm is not None and self._ok(self.video_label):
            try:
                rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb).resize((480, 360))
                imgtk = ImageTk.PhotoImage(img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            except Exception:
                pass
        self._refresh()
        self.root.after(50, self._tick)

    def _ok(self, w):
        try:
            return w is not None and w.winfo_exists()
        except tk.TclError:
            return False

    def _refresh(self):
        ratio = self.face_frames / max(self.total_frames, 1)
        el = max(time.time() - self.track_start, 1)
        bpm = self.blink_count / el * 60
        if self._ok(self.lbl_face):
            if self.face_ok:
                self.lbl_face.config(text=f"👤 Face: DETECTED ✓  ({ratio*100:.0f}%)", fg='green')
            else:
                self.lbl_face.config(text=f"👤 Face: NOT DETECTED ✗  ({ratio*100:.0f}%)", fg='red')
        if self._ok(self.lbl_blink):
            self.lbl_blink.config(text=f"👁️ Blinks: {self.blink_count}  ({bpm:.0f}/min)")
        if self._ok(self.lbl_eye):
            self.lbl_eye.config(text=f"👁️ Eyes: {self.eye_str}")
        if self._ok(self.lbl_head):
            hx, hy = self.head_offset
            d = "Centered"
            if abs(hx) > 0.2:
                d = "Left" if hx < 0 else "Right"
            elif abs(hy) > 0.15:
                d = "Down" if hy > 0 else "Up"
            self.lbl_head.config(text=f"🧠 Head: {d}  (x={hx:+.2f}  y={hy:+.2f})")
        if self._ok(self.lbl_cursor):
            self.lbl_cursor.config(text=f"🖱️ Speed: {self.cur_speed:.0f} px/s  |  Jitter: {self.cur_jitter:.0f}")
        if self._ok(self.lbl_idle):
            self.lbl_idle.config(text=f"⏱️ Cursor Idle: {self.cur_idle:.1f}s")

    def _clear(self):
        for w in self.root.winfo_children():
            w.destroy()
        self.video_label = None
        self.lbl_face = self.lbl_blink = self.lbl_eye = None
        self.lbl_head = self.lbl_cursor = self.lbl_idle = None

    def _metrics_panel(self, parent):
        c = THEMES[self.theme]
        fr = tk.Frame(parent, bg=c['card_bg'])
        fr.pack(fill=tk.X, padx=10, pady=5)
        kw = dict(font=("Arial", 10), bg=c['card_bg'], fg=c['card_fg'], anchor='w')
        self.lbl_face = tk.Label(fr, text="👤 Face: …", **kw); self.lbl_face.pack(fill=tk.X, pady=1)
        self.lbl_blink = tk.Label(fr, text="👁️ Blinks: …", **kw); self.lbl_blink.pack(fill=tk.X, pady=1)
        self.lbl_eye = tk.Label(fr, text="👁️ Eyes: …", **kw); self.lbl_eye.pack(fill=tk.X, pady=1)
        self.lbl_head = tk.Label(fr, text="🧠 Head: …", **kw); self.lbl_head.pack(fill=tk.X, pady=1)
        self.lbl_cursor = tk.Label(fr, text="🖱️ Cursor: …", **kw); self.lbl_cursor.pack(fill=tk.X, pady=1)
        self.lbl_idle = tk.Label(fr, text="⏱️ Idle: …", **kw); self.lbl_idle.pack(fill=tk.X, pady=1)

    def _check(self, user, correct):
        if not correct:
            return True
        u = user.strip().lower()
        c = correct.strip().lower()
        try:
            return abs(float(u) - float(c)) < 0.01
        except ValueError:
            pass
        return c == u or c in u

    def _calc_load(self, rt, q):
        sc = 0.0
        reasons = []
        if not self.face_ok:
            sc += 0.8
            reasons.append("Face not detected — looking away or disengaged")
        else:
            if "closed" in self.eye_str.lower() or "hidden" in self.eye_str.lower():
                sc += 0.7
                reasons.append("Eyes closed / hidden — strain or deep thought")
            elif "one" in self.eye_str.lower():
                sc += 0.3
                reasons.append("One eye detected — squinting or partial occlusion")
            hx, hy = self.head_offset
            if abs(hx) > 0.25:
                sc += 0.4
                reasons.append(f"Head turned {'left' if hx < 0 else 'right'} — looking away")
            if abs(hy) > 0.2:
                sc += 0.3
                reasons.append(f"Head tilted {'down' if hy > 0 else 'up'} — fatigue / confusion")
        el = max(time.time() - self.track_start, 1)
        bpm = self.blink_count / el * 60
        if bpm > 28:
            sc += 0.4
            reasons.append(f"Elevated blink rate ({bpm:.0f}/min) — stress")
        elif bpm < 8 and el > 30:
            sc += 0.2
            reasons.append(f"Low blink rate ({bpm:.0f}/min) — intense staring")
        if rt > 12:
            sc += 0.6
            reasons.append(f"Very long response ({rt:.1f}s)")
        elif rt > 7:
            sc += 0.3
            reasons.append(f"Slow response ({rt:.1f}s)")
        elif rt < 1.5:
            sc += 0.2
            reasons.append(f"Very fast response ({rt:.1f}s) — possible guess")
        if self.cur_jitter > 80:
            sc += 0.4
            reasons.append(f"High cursor jitter ({self.cur_jitter:.0f})")
        elif self.cur_jitter > 40:
            sc += 0.2
            reasons.append(f"Moderate cursor jitter ({self.cur_jitter:.0f})")
        if self.cur_idle > 5:
            sc += 0.2
            reasons.append(f"Cursor idle {self.cur_idle:.1f}s — hesitation")
        sc += q['difficulty'] * 0.2
        reasons.append(f"Question difficulty: {['Easy','Medium','Hard'][q['difficulty']]}")
        return min(2.0, max(0.0, sc)), reasons

    def _build_login(self):
        self._clear()
        self.session_active = False
        self.camera_processing = True
        c = THEMES[self.theme]

        hdr = tk.Frame(self.root, bg=c['header_bg'], height=90)
        hdr.pack(fill=tk.X); hdr.pack_propagate(False)
        tk.Label(hdr, text="🧠 Multimodal Cognitive Load Detection",
                 font=("Arial", 20, "bold"), fg=c['header_fg'], bg=c['header_bg']).pack(pady=10)
        tk.Label(hdr, text="Live Face  |  M5 Late Fusion  |  ADHD & Dyslexia",
                 font=("Arial", 11), fg='#BDC3C7', bg=c['header_bg']).pack()
        tk.Button(hdr, text="🌙 Dark" if self.theme == 'light' else "☀️ Light",
                  command=self._toggle_theme, bg=c['accent'], fg='white'
                  ).place(relx=0.93, rely=0.5, anchor='center')

        body = tk.Frame(self.root, bg=c['bg'])
        body.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        left = tk.LabelFrame(body, text="📹 Live Face Detection",
                             font=("Arial", 12, "bold"), bg=c['card_bg'], fg=c['card_fg'])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        self.video_label = tk.Label(left, bg='black')
        self.video_label.pack(pady=10, padx=10)
        self._metrics_panel(left)

        right = tk.Frame(body, bg=c['card_bg'], relief=tk.RAISED, bd=1)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        tk.Label(right, text="Student Login", font=("Arial", 18, "bold"),
                 bg=c['card_bg'], fg=c['card_fg']).pack(pady=25)
        tk.Label(right, text="Name:", font=("Arial", 12), bg=c['card_bg'], fg=c['card_fg']).pack()
        self.name_entry = tk.Entry(right, font=("Arial", 14), width=25)
        self.name_entry.pack(pady=5)
        self.name_entry.bind('<Return>', lambda _: self._start())
        tk.Label(right, text="Age:", font=("Arial", 12), bg=c['card_bg'], fg=c['card_fg']).pack(pady=(15, 0))
        self.age_entry = tk.Entry(right, font=("Arial", 14), width=10)
        self.age_entry.pack(pady=5)
        self.age_entry.bind('<Return>', lambda _: self._start())
        tk.Button(right, text="🎯 Start Assessment", command=self._start,
                  bg=c['button_bg'], fg='white', font=("Arial", 13, "bold"),
                  padx=25, pady=10).pack(pady=25)
        tk.Button(right, text="📊 View History", command=self._history,
                  bg=c['accent'], fg='white', font=("Arial", 11), padx=20, pady=8).pack(pady=5)
        info = ("🔬 How It Works:\n\n"
                "1. Camera tracks face, eyes, blinks\n"
                "2. Cursor speed & jitter tracked\n"
                "3. Response time measured per question\n"
                "4. All fused → cognitive load estimate\n\n"
                "📊 Levels:\n"
                "  🟢 Low — relaxed, comfortable\n"
                "  🟡 Medium — thinking hard\n"
                "  🔴 High — struggling, stressed\n\n"
                "After 12 questions you get a full report.")
        tk.Label(right, text=info, font=("Arial", 10), bg=c['card_bg'],
                 fg=c['card_fg'], justify=tk.LEFT).pack(pady=15, padx=20)

    def _start(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Required", "Enter your name.")
            return
        age = self.age_entry.get().strip()
        if age and not age.isdigit():
            messagebox.showwarning("Invalid Age", "Age must be a number.")
            return
        self.current_student = name
        self.session_start = datetime.now()
        self.session_active = True
        self.camera_processing = True
        self.q_idx = 0
        self.responses = []
        self.load_history = []
        self.submitting_answer = False
        self.face_frames = 0
        self.total_frames = 0
        self.blink_count = 0
        self._eyes_open = True
        self._open_streak = 0
        self._last_blink_t = 0.0
        self.track_start = time.time()
        if self.mouse_tracker:
            self.mouse_tracker.reset()
        self.questions_list = QUESTIONS.copy()
        random.shuffle(self.questions_list)
        self._build_quiz()
        self._show_q()

    def _build_quiz(self):
        self._clear()
        c = THEMES[self.theme]
        n = len(self.questions_list)
        hdr = tk.Frame(self.root, bg=c['header_bg'], height=60)
        hdr.pack(fill=tk.X); hdr.pack_propagate(False)
        tk.Label(hdr, text=f"👤 {self.current_student}", font=("Arial", 12, "bold"),
                 fg=c['header_fg'], bg=c['header_bg']).pack(side=tk.LEFT, padx=20, pady=15)
        self.prog_lbl = tk.Label(hdr, font=("Arial", 12), fg=c['header_fg'], bg=c['header_bg'])
        self.prog_lbl.pack(side=tk.RIGHT, padx=20, pady=15)
        self.prog_bar = ttk.Progressbar(hdr, length=250, maximum=n)
        self.prog_bar.pack(side=tk.RIGHT, padx=10, pady=15)

        body = tk.Frame(self.root, bg=c['bg'])
        body.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        left = tk.LabelFrame(body, text="📹 Live Face Analysis", font=("Arial", 12, "bold"),
                             bg=c['card_bg'], fg=c['card_fg'])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.video_label = tk.Label(left, bg='black')
        self.video_label.pack(pady=10, padx=10)
        self._metrics_panel(left)

        right = tk.LabelFrame(body, text="📝 Question", font=("Arial", 12, "bold"),
                              bg=c['card_bg'], fg=c['card_fg'])
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.q_lbl = tk.Label(right, text="", font=("Arial", 15), wraplength=420,
                              justify=tk.CENTER, bg=c['card_bg'], fg=c['card_fg'])
        self.q_lbl.pack(pady=45, padx=20)
        self.ans_entry = tk.Entry(right, font=("Arial", 14), width=30)
        self.ans_entry.pack(pady=10)
        self.ans_entry.bind('<Return>', lambda _: self._submit())
        self.submit_btn = tk.Button(right, text="✅ Submit Answer", command=self._submit,
                                    bg=c['button_bg'], fg=c['button_fg'],
                                    font=("Arial", 12, "bold"), padx=20, pady=8)
        self.submit_btn.pack(pady=15)
        self.fb_lbl = tk.Label(right, text="", font=("Arial", 12), bg=c['card_bg'])
        self.fb_lbl.pack(pady=5)
        lf = tk.LabelFrame(right, text="Cognitive Load", font=("Arial", 10, "bold"),
                           bg=c['card_bg'], fg=c['card_fg'])
        lf.pack(fill=tk.X, padx=15, pady=10)
        self.load_lbl = tk.Label(lf, text="Awaiting answer…", font=("Arial", 16, "bold"),
                                 bg=c['card_bg'], fg='gray')
        self.load_lbl.pack(pady=10)

    def _show_q(self):
        if self.q_idx >= len(self.questions_list):
            return
        self.submitting_answer = False
        if self._ok(self.submit_btn):
            self.submit_btn.config(state=tk.NORMAL, text="✅ Submit Answer")
        if self._ok(self.ans_entry):
            self.ans_entry.config(state=tk.NORMAL)
        total = len(self.questions_list)
        q = self.questions_list[self.q_idx]
        self.q_lbl.config(text=f"Question {self.q_idx+1} of {total}\n\n{q['text']}")
        self.ans_entry.delete(0, tk.END)
        self.prog_lbl.config(text=f"Q{self.q_idx+1}/{total}")
        self.prog_bar['value'] = self.q_idx
        self.q_start = time.time()
        self.fb_lbl.config(text="")
        self.load_lbl.config(text="Awaiting answer…", fg='gray')
        self.ans_entry.focus_set()

    def _submit(self):
        if self.submitting_answer:
            return
        if not self._ok(self.ans_entry):
            return
        if self.q_idx >= len(self.questions_list):
            return
        user_answer = self.ans_entry.get().strip()
        if not user_answer:
            self.fb_lbl.config(text="⚠️ Type an answer before submitting.", fg="red")
            return
        self.submitting_answer = True
        if self._ok(self.submit_btn):
            self.submit_btn.config(state=tk.DISABLED, text="Answer Submitted")
        if self._ok(self.ans_entry):
            self.ans_entry.config(state=tk.DISABLED)
        response_time = time.time() - self.q_start
        question = self.questions_list[self.q_idx]
        is_correct = self._check(user_answer, question["answer"])
        cognitive_load, load_reasons = self._calc_load(response_time, question)
        load_index = max(0, min(2, int(round(cognitive_load))))
        load_text = LOAD_LABELS[load_index]
        face_ratio = self.face_frames / max(self.total_frames, 1)
        session_elapsed = max(time.time() - self.track_start, 1)
        blink_rate = (self.blink_count / session_elapsed) * 60
        eyes_not_visible = ("closed" in self.eye_str.lower() or "hidden" in self.eye_str.lower()
                            or "no face" in self.eye_str.lower())
        face_features = {
            "eye_closure": 0.7 if eyes_not_visible else 0.2,
            "gaze_stability": 0.8 if self.face_ok else 0.3,
            "head_stability": max(0.0, 1.0 - abs(self.head_offset[0])),
            "pose_Rx": self.head_offset[1], "pose_Ry": self.head_offset[0],
            "pose_Rz": 0.0, "face_detected": int(self.face_ok), "blink_rate": blink_rate
        }
        if self.mouse_tracker:
            cursor_features = self.mouse_tracker.get_features()
        else:
            cursor_features = [self.cur_speed, self.cur_jitter, 0.0, 0.0, self.cur_idle, 0.0, 0.0, 0.0]
        interaction_features = [response_time, 0.0, 0.0, 0.0, 0.0, 0.0, response_time]
        if self.model_predictor:
            model_predictions = self.model_predictor.get_all_predictions(
                face_features, cursor_features, interaction_features,
                response_time, question["difficulty"], self.face_ok)
        else:
            fc = 0.5
            model_predictions = {
                "M1: FAU": {"level": load_index, "confidence": fc},
                "M2: Cursor": {"level": load_index, "confidence": fc},
                "M3: Interaction": {"level": load_index, "confidence": fc},
                "M4: Early": {"level": load_index, "confidence": fc},
                "M5: Late⭐": {"level": load_index, "confidence": fc,
                              "modality_weights": {"facial": 0.45, "cursor": 0.30, "interaction": 0.25}}
            }
        self.responses.append({
            "question": question["text"], "type": question["type"],
            "difficulty": question["difficulty"], "user_answer": user_answer,
            "correct_answer": question["answer"], "correct": is_correct,
            "time": response_time, "load": cognitive_load, "load_txt": load_text,
            "face_ok": self.face_ok, "face_ratio": face_ratio, "blink_rate": blink_rate,
            "eye_status": self.eye_str, "head": self.head_offset,
            "cur_speed": self.cur_speed, "cur_jitter": self.cur_jitter,
            "cur_idle": self.cur_idle, "reasons": load_reasons,
            "model_predictions": model_predictions
        })
        self.load_history.append(cognitive_load)
        if is_correct:
            self.fb_lbl.config(text=f"✅ Correct!  |  {response_time:.1f} seconds", fg="#4CAF50")
        else:
            self.fb_lbl.config(text=f"❌ Incorrect  |  Correct answer: {question['answer']}  |  {response_time:.1f} seconds", fg="#F44336")
        self.load_lbl.config(text=f"Cognitive Load: {load_text}", fg=LOAD_COLORS[load_text])
        self.q_idx += 1
        if self.q_idx < len(self.questions_list):
            self.root.after(1200, self._show_q)
        else:
            self.prog_bar["value"] = len(self.questions_list)
            self.fb_lbl.config(text="✅ Assessment completed. Preparing analysis…", fg="#2196F3")
            self.root.after(1200, self._finish)

    def _finish(self):
        self.session_active = False
        self.camera_processing = False
        total = len(self.responses)
        if total == 0:
            self._build_login()
            return
        correct = sum(1 for r in self.responses if r["correct"])
        accuracy = correct / total
        average_load = sum(r["load"] for r in self.responses) / total
        average_response_time = sum(r["time"] for r in self.responses) / total
        average_face_ratio = self.face_frames / max(self.total_frames, 1)
        tracking_duration = max(time.time() - self.track_start, 1)
        final_blink_rate = (self.blink_count / tracking_duration) * 60
        average_cursor_speed = sum(r["cur_speed"] for r in self.responses) / total
        average_cursor_jitter = sum(r["cur_jitter"] for r in self.responses) / total
        high_load_count = sum(1 for r in self.responses if r["load"] > 1.4)
        model_names = ["M1: FAU", "M2: Cursor", "M3: Interaction", "M4: Early", "M5: Late⭐"]
        model_summary = {}
        for model_name in model_names:
            levels, confidences = [], []
            for r in self.responses:
                p = r.get("model_predictions", {}).get(model_name)
                if p:
                    levels.append(p.get("level", 1))
                    confidences.append(p.get("confidence", 0.5))
            if levels:
                aml = sum(levels) / len(levels)
                ac = sum(confidences) / len(confidences)
                lc = sum(1 for lv in levels if lv == 0)
                mc = sum(1 for lv in levels if lv == 1)
                hc = sum(1 for lv in levels if lv == 2)
            else:
                aml, ac, lc, mc, hc = average_load, 0.5, 0, total, 0
            model_summary[model_name] = {"average_level": aml, "confidence": ac,
                                         "low_count": lc, "medium_count": mc, "high_count": hc}
        mws = []
        for r in self.responses:
            w = r.get("model_predictions", {}).get("M5: Late⭐", {}).get("modality_weights")
            if w:
                mws.append(w)
        if mws:
            modality_weights = {
                "facial": sum(w.get("facial", 0) for w in mws) / len(mws),
                "cursor": sum(w.get("cursor", 0) for w in mws) / len(mws),
                "interaction": sum(w.get("interaction", 0) for w in mws) / len(mws)
            }
        else:
            modality_weights = {"facial": 0.45, "cursor": 0.30, "interaction": 0.25}
        adhd_risk, dyslexia_risk = 0.1, 0.1
        if average_face_ratio < 0.70:
            adhd_risk += 0.30
        if final_blink_rate > 25:
            adhd_risk += 0.20; dyslexia_risk += 0.15
        if average_cursor_jitter > 60:
            adhd_risk += 0.20
        if high_load_count > total * 0.5:
            adhd_risk += 0.10; dyslexia_risk += 0.10
        slow = sum(1 for r in self.responses if r["time"] > 10)
        if slow > total * 0.4:
            dyslexia_risk += 0.20
        adhd_risk = min(0.95, adhd_risk)
        dyslexia_risk = min(0.95, dyslexia_risk)
        session_duration = (datetime.now() - self.session_start).total_seconds()
        if self.db:
            try:
                sid = self.db.get_or_create_student(self.current_student)
                ssid = self.db.save_session(sid, average_load, accuracy, adhd_risk, dyslexia_risk)
                for r in self.responses:
                    self.db.save_response(ssid, r["question"], r["type"], r["difficulty"],
                                          int(round(r["load"])), r["time"], r["correct"])
            except Exception as e:
                print(f"Database warning: {e}")
        results = {
            "total": total, "correct": correct, "acc": accuracy,
            "avg_load": average_load, "avg_time": average_response_time,
            "avg_face": average_face_ratio, "avg_blink": final_blink_rate,
            "total_blinks": self.blink_count, "avg_speed": average_cursor_speed,
            "avg_jitter": average_cursor_jitter, "adhd": adhd_risk, "dys": dyslexia_risk,
            "hlc": high_load_count, "duration": session_duration,
            "model_summary": model_summary, "modality_weights": modality_weights
        }
        self._build_analysis(results)

    def _build_analysis(self, results):
        self._clear()
        self.camera_processing = False
        colors = THEMES[self.theme]
        header = tk.Frame(self.root, bg=colors["header_bg"], height=70)
        header.pack(fill=tk.X); header.pack_propagate(False)
        tk.Label(header, text="📊 Student Performance & Cognitive Load Analysis",
                 font=("Arial", 19, "bold"), fg=colors["header_fg"], bg=colors["header_bg"]).pack(pady=17)
        main_container = tk.Frame(self.root, bg=colors["bg"])
        main_container.pack(fill=tk.BOTH, expand=True, padx=35, pady=20)
        report_frame = tk.LabelFrame(main_container, text="Complete Session Report",
                                     font=("Arial", 13, "bold"), bg=colors["card_bg"], fg=colors["card_fg"])
        report_frame.pack(fill=tk.BOTH, expand=True)
        report_text = scrolledtext.ScrolledText(report_frame, font=("Consolas", 10),
                                                bg=colors["card_bg"], fg=colors["card_fg"],
                                                wrap=tk.WORD, padx=15, pady=15)
        report_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        report_text.insert(tk.END, self._report(results))
        report_text.config(state=tk.DISABLED)
        button_frame = tk.Frame(main_container, bg=colors["bg"])
        button_frame.pack(pady=15)
        tk.Button(button_frame, text="🏠 Return Home", command=self._build_login,
                  bg=colors["button_bg"], fg="white", font=("Arial", 12, "bold"),
                  padx=25, pady=8).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="📁 View History", command=self._history,
                  bg=colors["accent"], fg="white", font=("Arial", 12),
                  padx=25, pady=8).pack(side=tk.LEFT, padx=10)

    def _report(self, results):
        ali = max(0, min(2, int(round(results["avg_load"]))))
        alt = LOAD_LABELS[ali]
        low = sum(1 for r in self.responses if r["load"] < 0.7)
        med = sum(1 for r in self.responses if 0.7 <= r["load"] <= 1.4)
        high = sum(1 for r in self.responses if r["load"] > 1.4)
        dm = results["duration"] / 60
        L = []
        L.append("=" * 76)
        L.append("              COGNITIVE LOAD ANALYSIS REPORT")
        L.append("=" * 76)
        L.append("")
        L.append(f"Student:              {self.current_student}")
        L.append(f"Assessment Date:      {self.session_start.strftime('%Y-%m-%d %H:%M')}")
        L.append(f"Assessment Duration:  {dm:.1f} minutes")
        L.append("")
        L.append("-" * 76)
        L.append("1. PERFORMANCE SUMMARY")
        L.append("-" * 76)
        L.append(f"Correct Answers:       {results['correct']}/{results['total']}")
        L.append(f"Accuracy:              {results['acc']*100:.1f}%")
        L.append(f"Average Response Time: {results['avg_time']:.1f} seconds")
        L.append(f"Average Load:          {alt} ({results['avg_load']:.2f}/2.00)")
        if results["acc"] >= 0.85:
            L.append("Performance Summary:  Excellent overall performance.")
        elif results["acc"] >= 0.65:
            L.append("Performance Summary:  Good performance with some areas for improvement.")
        elif results["acc"] >= 0.45:
            L.append("Performance Summary:  Moderate performance. Additional practice may help.")
        else:
            L.append("Performance Summary:  The student experienced difficulty with several questions.")
        L.append("")
        L.append("Cognitive Load Distribution:")
        L.append(f"  Low:                 {low} questions")
        L.append(f"  Medium:              {med} questions")
        L.append(f"  High:                {high} questions")
        L.append("")
        L.append("-" * 76)
        L.append("2. FIVE-MODEL COGNITIVE LOAD ANALYSIS")
        L.append("-" * 76)
        md = {
            "M1: FAU": "Facial model: uses face visibility, eye state and head-position information.",
            "M2: Cursor": "Cursor model: uses cursor speed, movement variation, idle time and path behavior.",
            "M3: Interaction": "Interaction model: uses response time and quiz interaction behavior.",
            "M4: Early": "Early-fusion model: combines all raw modality features before estimating load.",
            "M5: Late⭐": "Late-fusion model: combines M1, M2 and M3 predictions using dynamic weights."
        }
        for mn, s in results["model_summary"].items():
            mli = max(0, min(2, int(round(s["average_level"]))))
            L.append("")
            L.append(f"{mn}")
            L.append(f"  Average Prediction:  {LOAD_LABELS[mli]} ({s['average_level']:.2f}/2.00)")
            L.append(f"  Average Confidence:  {s['confidence']*100:.1f}%")
            L.append(f"  Distribution:        Low {s['low_count']} | Medium {s['medium_count']} | High {s['high_count']}")
            L.append(f"  Interpretation:      {md.get(mn, '')}")
        w = results["modality_weights"]
        L.append("")
        L.append("M5 Average Modality Contribution:")
        L.append(f"  Facial contribution:      {w['facial']*100:.1f}%")
        L.append(f"  Cursor contribution:      {w['cursor']*100:.1f}%")
        L.append(f"  Interaction contribution: {w['interaction']*100:.1f}%")
        L.append("")
        L.append("-" * 76)
        L.append("3. FACIAL AND BLINK ANALYSIS")
        L.append("-" * 76)
        fp = results["avg_face"] * 100
        L.append(f"Face Detection Rate:   {fp:.1f}%")
        L.append(f"Total Blinks:          {results['total_blinks']}")
        L.append(f"Average Blink Rate:    {results['avg_blink']:.1f} blinks/minute")
        if fp >= 85:
            L.append("Face Engagement:       Strong and consistent screen engagement.")
        elif fp >= 65:
            L.append("Face Engagement:       Moderate engagement with some looking away.")
        else:
            L.append("Face Engagement:       Face was frequently not visible.")
        if results["avg_blink"] < 8:
            L.append("Blink Interpretation:  Low blink rate, possibly intense visual focus.")
        elif results["avg_blink"] <= 25:
            L.append("Blink Interpretation:  Blink rate within a broadly typical range.")
        else:
            L.append("Blink Interpretation:  Elevated blink rate; may reflect fatigue or stress.")
        ahh = sum(abs(r["head"][0]) for r in self.responses) / results["total"]
        avh = sum(abs(r["head"][1]) for r in self.responses) / results["total"]
        L.append(f"Average Head Offset:   horizontal {ahh:.2f}, vertical {avh:.2f}")
        if ahh < 0.15 and avh < 0.15:
            L.append("Head Movement:         Mostly stable and centered.")
        elif ahh < 0.30:
            L.append("Head Movement:         Some movement was observed.")
        else:
            L.append("Head Movement:         Frequent movement or looking away observed.")
        L.append("")
        L.append("-" * 76)
        L.append("4. CURSOR BEHAVIOR ANALYSIS")
        L.append("-" * 76)
        L.append(f"Average Cursor Speed:  {results['avg_speed']:.1f} pixels/second")
        L.append(f"Average Cursor Jitter: {results['avg_jitter']:.1f}")
        if results["avg_jitter"] < 25:
            L.append("Cursor Interpretation: Movement was generally calm and steady.")
        elif results["avg_jitter"] < 60:
            L.append("Cursor Interpretation: Moderate movement variation was observed.")
        else:
            L.append("Cursor Interpretation: High variation; may reflect uncertainty or restlessness.")
        L.append("")
        L.append("-" * 76)
        L.append("5. PERFORMANCE BY QUESTION DIFFICULTY")
        L.append("-" * 76)
        dn = ["Easy", "Medium", "Hard"]
        for dv, dnm in enumerate(dn):
            dr = [r for r in self.responses if r["difficulty"] == dv]
            dt = len(dr)
            dc = sum(1 for r in dr if r["correct"])
            dat = (sum(r["time"] for r in dr) / dt) if dt else 0
            dal = (sum(r["load"] for r in dr) / dt) if dt else 0
            L.append("")
            L.append(f"{dnm} Questions:")
            L.append(f"  Correct:              {dc}/{dt}")
            L.append(f"  Average Time:         {dat:.1f} seconds")
            L.append(f"  Average Load:         {dal:.2f}/2.00")
        L.append("")
        L.append("-" * 76)
        L.append("6. QUESTION-BY-QUESTION RESULTS")
        L.append("-" * 76)
        for i, r in enumerate(self.responses, start=1):
            status = "CORRECT" if r["correct"] else "INCORRECT"
            L.append("")
            L.append(f"Question {i}: {r['question']}")
            L.append(f"  Student Answer:      {r['user_answer']}")
            L.append(f"  Correct Answer:      {r['correct_answer']}")
            L.append(f"  Result:              {status}")
            L.append(f"  Difficulty:          {dn[r['difficulty']]}")
            L.append(f"  Response Time:       {r['time']:.1f} seconds")
            L.append(f"  Cognitive Load:      {r['load_txt']}")
        L.append("")
        L.append("-" * 76)
        L.append("7. LEARNING-SUPPORT INDICATORS")
        L.append("-" * 76)
        al = "Low" if results["adhd"] < 0.33 else "Moderate" if results["adhd"] < 0.66 else "High"
        dl = "Low" if results["dys"] < 0.33 else "Moderate" if results["dys"] < 0.66 else "High"
        L.append(f"Attention-support indicator: {results['adhd']*100:.1f}% ({al})")
        L.append(f"Reading-support indicator:   {results['dys']*100:.1f}% ({dl})")
        L.append("")
        L.append("Important: These are behavioral support indicators only.")
        L.append("They are not medical diagnoses of ADHD or dyslexia.")
        L.append("")
        L.append("-" * 76)
        L.append("8. RECOMMENDATIONS")
        L.append("-" * 76)
        recs = []
        if results["avg_load"] > 1.4:
            recs.append("Use shorter learning blocks with regular breaks.")
            recs.append("Divide complex questions into smaller steps.")
        elif results["avg_load"] > 0.7:
            recs.append("Continue using structured tasks and clear progress indicators.")
        else:
            recs.append("The student appeared comfortable with the current task format.")
        if results["avg_time"] > 10:
            recs.append("Allow additional time for analytical questions.")
        if results["avg_face"] < 0.70:
            recs.append("Reduce nearby visual distractions and improve screen positioning.")
        if results["avg_blink"] > 25:
            recs.append("Encourage short visual breaks and comfortable screen lighting.")
        if results["avg_jitter"] > 60:
            recs.append("Use larger buttons and simpler screen layouts.")
        if results["acc"] < 0.60:
            recs.append("Review missed concepts and begin with easier examples.")
        elif results["acc"] >= 0.85:
            recs.append("Consider introducing slightly more challenging questions.")
        for rec in recs:
            L.append(f"• {rec}")
        L.append("")
        L.append("=" * 76)
        L.append("                         END OF REPORT")
        L.append("=" * 76)
        return "\n".join(L)

    def _history(self):
        self._clear()
        c = THEMES[self.theme]
        hdr = tk.Frame(self.root, bg=c["header_bg"], height=55)
        hdr.pack(fill=tk.X); hdr.pack_propagate(False)
        tk.Label(hdr, text="📁 Session History", font=("Arial", 18, "bold"),
                 fg=c["header_fg"], bg=c["header_bg"]).pack(pady=12)
        txt = scrolledtext.ScrolledText(self.root, font=("Consolas", 10),
                                        bg=c["card_bg"], fg=c["card_fg"], wrap=tk.WORD)
        txt.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        if self.responses:
            nc = sum(1 for r in self.responses if r["correct"])
            txt.insert(tk.END, f"Latest Session — {self.current_student}\n")
            txt.insert(tk.END, "=" * 60 + "\n")
            txt.insert(tk.END, f"Score: {nc}/{len(self.responses)} ({nc/len(self.responses)*100:.1f}%)\n\n")
            for i, r in enumerate(self.responses, start=1):
                mark = "✅" if r["correct"] else "❌"
                txt.insert(tk.END, f"Q{i}: {r['question'][:55]}\n")
                txt.insert(tk.END, f"   {mark}  |  Load: {r['load_txt']}  |  {r['time']:.1f}s\n\n")
        if self.db and self.current_student:
            sessions = self.db.get_student_sessions(self.current_student)
            if sessions:
                txt.insert(tk.END, "\n" + "=" * 60 + "\n")
                txt.insert(tk.END, "DATABASE SESSIONS:\n\n")
                for s in sessions:
                    txt.insert(tk.END, f"  {s[1]}  |  Load: {s[2]:.2f}  |  Acc: {s[3]*100:.0f}%  |  "
                                       f"ADHD: {s[4]*100:.0f}%  |  Dys: {s[5]*100:.0f}%\n")
        if not self.responses:
            txt.insert(tk.END, "No data yet. Complete an assessment first.\n")
        txt.config(state=tk.DISABLED)
        tk.Button(self.root, text="← Back", command=self._build_login,
                  bg=c["accent"], fg="white", font=("Arial", 12), padx=20).pack(pady=10)

    def _toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        if self.session_active:
            idx = self.q_idx
            self._build_quiz()
            self.q_idx = idx
            self._show_q()
        else:
            self._build_login()

    def on_closing(self):
        self.is_running = False
        time.sleep(0.15)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = CognitiveLoadApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()