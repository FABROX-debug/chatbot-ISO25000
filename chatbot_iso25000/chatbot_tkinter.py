import os, json, time
import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ===================== CONFIG =====================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CSV_FILE      = "dataset_ragas_definitivo.csv"
EMB_MODEL     = "text-embedding-3-small"
EMB_FILE      = "embeddings.npy"
CUTOFF_CHARS  = 6000

# ===================== PROMPT MEJORADO =====================
PROMPT_TEMPLATE = """
Eres “ISO Insights”, un asistente experto en la familia de normas ISO/IEC 25000 (SQuaRE):
ISO/IEC 25000, 25010, 25012, 25022, 25023, 25024 y 25040.

Comportamiento:
- Si el mensaje es un SALUDO, despedida o agradecimiento, responde cordialmente y ofrece ayuda sobre ISO/IEC 25000.
- Si el texto menciona “normas ISO” sin especificar, asume que se refiere a la familia ISO/IEC 25000.
- Si menciona otra familia (9001, 27001, 14000...), responde amablemente que tu especialidad es ISO/IEC 25000, y ofrece temas afines (modelo de calidad, métricas, evaluación de producto, etc.).
- Si la pregunta está fuera del tema, di brevemente:  
  “Puedo ayudarte con temas sobre calidad del software según ISO/IEC 25000 (25010, 25012, 25022, 25023, 25024, 25040).”
- Usa el CONTEXTO recuperado como apoyo real. No inventes.

Formato de respuesta:
- En español, claro y profesional (3–6 líneas máximo).
- Usa viñetas o frases cortas si ayuda a la comprensión.
- Menciona la norma exacta cuando corresponda (p. ej., “modelo de calidad (ISO/IEC 25010)”).

CONTEXTO:
{contexto}

PREGUNTA:
{pregunta}
"""

# ===================== COLORES =====================
C = {
    "bg": "#EAF2FB",
    "header": "#E8F0FA",
    "title": "#0F2B46",
    "panel": "#EAF2FB",
    "bot_bubble": "#FFFFFF",
    "user_bubble": "#2F6DB3",
    "user_text": "#FFFFFF",
    "bot_text": "#0F2B46",
    "input_bg": "#FFFFFF",
    "send_bg": "#2F6DB3",
    "send_fg": "#FFFFFF",
    "muted": "#7A8AA0"
}

# ===================== FUNCIONES =====================
def embed_texts(texts):
    vecs = []
    for i in range(0, len(texts), 50):
        batch = [t[:CUTOFF_CHARS] for t in texts[i:i + 50]]
        resp = client.embeddings.create(model=EMB_MODEL, input=batch)
        vecs.extend([d.embedding for d in resp.data])
        print(f"Embeddings {i+len(batch)}/{len(texts)}")
    return np.array(vecs, dtype=np.float32)

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T

# ===================== CARGA DATA =====================
print("📂 Cargando dataset...")
df = pd.read_csv(CSV_FILE).dropna(subset=["context", "ground_truth"])
contexts = df["context"].astype(str).tolist()
answers  = df["ground_truth"].astype(str).tolist()

if os.path.exists(EMB_FILE):
    context_embeddings = np.load(EMB_FILE)
    print("✅ Embeddings cargados.")
else:
    context_embeddings = embed_texts(contexts)
    np.save(EMB_FILE, context_embeddings)

# ===================== APP =====================
class ChatBotApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ISO Insights")
        self.geometry("920x680")
        self.configure(bg=C["bg"])
        self._header()
        self._body()
        self._input_bar()
        self._bot_message("Hola, soy ISO Insights. ¿En qué puedo ayudarte hoy con las normas ISO/IEC 25000?")

    # -------- HEADER ----------
    def _header(self):
        h = tk.Frame(self, bg=C["header"], height=60)
        h.pack(fill="x")
        logo = tk.Label(h, text="📘", bg=C["header"], font=("Segoe UI Emoji", 16))
        logo.pack(side="left", padx=15)
        tk.Label(h, text="ISO Insights", bg=C["header"], fg=C["title"], font=("Segoe UI Semibold", 18)).pack(side="left")

    # -------- BODY ----------
    def _body(self):
        f = tk.Frame(self, bg=C["panel"])
        f.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(f, bg=C["panel"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(f, orient="vertical", command=self.canvas.yview)
        self.chat_frame = tk.Frame(self.canvas, bg=C["panel"])
        self.canvas.create_window((0, 0), window=self.chat_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        self.chat_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    # -------- INPUT BAR ----------
    def _input_bar(self):
        bar = tk.Frame(self, bg=C["header"])
        bar.pack(fill="x", side="bottom")
        self.entry = tk.Entry(bar, font=("Segoe UI", 12), bg=C["input_bg"], fg=C["title"])
        self.entry.pack(side="left", fill="x", expand=True, padx=15, pady=10, ipady=8)
        self.entry.insert(0, "Pregunta sobre las normas ISO...")
        self.entry.bind("<Return>", lambda e: self._send_message())
        send = tk.Button(bar, text="➤", font=("Segoe UI", 11, "bold"), bg=C["send_bg"], fg=C["send_fg"],
                         relief="flat", width=4, cursor="hand2", command=self._send_message)
        send.pack(side="right", padx=10, pady=8)

    # -------- MENSAJES ----------
    def _message_bubble(self, text, side="left", color_bg="#FFF", color_fg="#000", emoji="🤖"):
        frame = tk.Frame(self.chat_frame, bg=C["panel"])
        frame.pack(fill="x", pady=6, padx=10, anchor="w" if side=="left" else "e")

        avatar = tk.Label(frame, text=emoji, font=("Segoe UI Emoji", 14), bg=C["panel"])
        avatar.pack(side="left" if side=="left" else "right", padx=(4, 10))

        msg = tk.Label(frame, text=text, wraplength=600, justify="left",
                       bg=color_bg, fg=color_fg, font=("Segoe UI", 11), padx=12, pady=8)
        msg.pack(side="left" if side=="left" else "right", padx=4)
        msg.configure(relief="flat", anchor="w", bd=0)
        msg.after(50, lambda: self.canvas.yview_moveto(1))

    def _bot_message(self, text):
        self._message_bubble(text, side="left", color_bg=C["bot_bubble"], color_fg=C["bot_text"], emoji="🤖")

    def _user_message(self, text):
        self._message_bubble(text, side="right", color_bg=C["user_bubble"], color_fg=C["user_text"], emoji="👤")

    # -------- ENVIAR ----------
    def _send_message(self):
        q = self.entry.get().strip()
        if not q or q == "Pregunta sobre las normas ISO...":
            return
        self.entry.delete(0, "end")
        self._user_message(q)
        self.after(100, lambda: self._generate_answer(q))

    # -------- RAG + LLM ----------
    def _generate_answer(self, question):
        try:
            q_vec = np.array(client.embeddings.create(model=EMB_MODEL, input=[question]).data[0].embedding)[None, :]
            sims = cosine_sim(q_vec, context_embeddings)[0]
            idx = int(np.argmax(sims))
            contexto = contexts[idx][:CUTOFF_CHARS]
        except Exception as e:
            self._bot_message("Error generando embeddings: " + str(e))
            return

        prompt = PROMPT_TEMPLATE.format(contexto=contexto, pregunta=question)

        try:
            comp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=350
            )
            ans = comp.choices[0].message.content.strip()
        except Exception as e:
            ans = "Error al generar respuesta: " + str(e)

        self._bot_message(ans)

# ===================== RUN =====================
if __name__ == "__main__":
    ChatBotApp().mainloop()
