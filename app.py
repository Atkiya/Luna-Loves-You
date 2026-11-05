"""Luna Loves You - tiny CPU deployment for Render's free tier (llama.cpp + Flask, no torch, no Gradio).

Env vars:
  GGUF_REPO   Hugging Face repo with the quantized model, e.g. "your-username/luna-loves-you-gguf"  (required)
  GGUF_FILE   file name in that repo (default luna-0.5b-Q4_K_M.gguf; use luna-0.5b-Q3_K_M.gguf if you run out of memory)
  HF_TOKEN    only if the repo is private
  MAX_NEW_TOKENS (default 160)   N_CTX (default 768)   N_THREADS (default 1)
"""
import os
import threading
from flask import Flask, Response, jsonify, request, stream_with_context

GGUF_REPO = os.environ.get("GGUF_REPO", "")
GGUF_FILE = os.environ.get("GGUF_FILE", "luna-0.5b-Q4_K_M.gguf")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "160"))
N_CTX = int(os.environ.get("N_CTX", "768"))
N_THREADS = int(os.environ.get("N_THREADS", "1"))

SYSTEM_PROMPT = (
    "You are a menstrual health education assistant. Give clear, simple, supportive, "
    "general information. For severe pain, very heavy bleeding, missed periods for three "
    "months or more, possible pregnancy, or any worrying symptom, advise seeing a doctor. "
    "If you are not sure, say so."
)

app = Flask(__name__)
state = {"llm": None, "error": None}
gen_lock = threading.Lock()          # one generation at a time: the free tier has almost no CPU or RAM


def load_model():
    """Runs in a background thread so the port opens immediately (Render's health check)."""
    try:
        if not GGUF_REPO:
            raise RuntimeError("Set the GGUF_REPO environment variable.")
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        path = hf_hub_download(GGUF_REPO, GGUF_FILE, local_dir="/tmp/model",
                               token=os.environ.get("HF_TOKEN") or None)
        state["llm"] = Llama(model_path=path, n_ctx=N_CTX, n_threads=N_THREADS,
                             n_batch=64, use_mmap=True, verbose=False)
    except Exception as e:           # show the reason in the chat instead of silently failing
        state["error"] = f"{type(e).__name__}: {e}"


threading.Thread(target=load_model, daemon=True).start()

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Luna Loves You</title>
<style>
  :root { --bg:#fff7fa; --card:#ffffff; --ink:#2b2230; --muted:#7a6b80; --accent:#c2477a; --user:#f6dbe8; --bot:#f1eef6; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#1d1822; --card:#272030; --ink:#f1eaf5; --muted:#a898b0; --accent:#e06a9c; --user:#4a2c3d; --bot:#332b3b; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, sans-serif; background:var(--bg); color:var(--ink);
         display:flex; justify-content:center; min-height:100vh; }
  main { width:100%; max-width:720px; padding:16px; display:flex; flex-direction:column; gap:12px; }
  h1 { margin:8px 0 0; color:var(--accent); font-size:1.5rem; }
  p.note { margin:0; color:var(--muted); font-size:.85rem; }
  #log { flex:1; background:var(--card); border-radius:14px; padding:12px; overflow-y:auto; min-height:50vh;
         display:flex; flex-direction:column; gap:10px; }
  .msg { padding:10px 12px; border-radius:12px; max-width:88%; white-space:pre-wrap; line-height:1.4; }
  .user { background:var(--user); align-self:flex-end; }
  .bot { background:var(--bot); align-self:flex-start; }
  form { display:flex; gap:8px; }
  input { flex:1; padding:12px; border-radius:10px; border:1px solid var(--muted); background:var(--card); color:var(--ink); font-size:1rem; }
  button { padding:12px 18px; border:0; border-radius:10px; background:var(--accent); color:#fff; font-size:1rem; cursor:pointer; }
  button:disabled { opacity:.5; cursor:wait; }
</style></head>
<body><main>
  <h1>Luna Loves You</h1>
  <p class="note">General education only. This is not medical advice. See a doctor for worrying symptoms.
  The first answer can take a while (the model wakes up and loads).</p>
  <div id="log"></div>
  <form id="f"><input id="q" placeholder="Ask about periods, cramps, cycles..." autocomplete="off" required>
  <button id="b">Send</button></form>
</main>
<script>
const log = document.getElementById('log'), f = document.getElementById('f'),
      q = document.getElementById('q'), b = document.getElementById('b');
function add(cls, text) {
  const d = document.createElement('div'); d.className = 'msg ' + cls; d.textContent = text;
  log.appendChild(d); log.scrollTop = log.scrollHeight; return d;
}
f.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = q.value.trim(); if (!text) return;
  q.value = ''; b.disabled = true; add('user', text);
  const out = add('bot', '...');
  try {
    const r = await fetch('/chat', { method: 'POST', headers: {'Content-Type': 'application/json'},
                                     body: JSON.stringify({message: text}) });
    if (!r.ok || !r.body) { out.textContent = (await r.text()) || 'Something went wrong.'; }
    else {
      const reader = r.body.getReader(), dec = new TextDecoder(); out.textContent = '';
      for (;;) { const {done, value} = await reader.read(); if (done) break;
                 out.textContent += dec.decode(value, {stream: true}); log.scrollTop = log.scrollHeight; }
    }
  } catch (err) { out.textContent = 'Network error. Please try again.'; }
  b.disabled = false; q.focus();
});
</script></body></html>"""


@app.get("/")
def index():
    return PAGE


@app.get("/health")
def health():
    return jsonify(ok=True, model_loaded=state["llm"] is not None, error=state["error"])


@app.post("/chat")
def chat():
    msg = ((request.get_json(silent=True) or {}).get("message") or "").strip()
    if not msg:
        return Response("Please type a question.", status=400, mimetype="text/plain")
    msg = msg[:600]
    if state["error"]:
        return Response("The model failed to load: " + state["error"], status=500, mimetype="text/plain")
    if state["llm"] is None:
        return Response("The model is still loading, please try again in a minute.", status=503, mimetype="text/plain")
    if not gen_lock.acquire(blocking=False):
        return Response("Luna is busy with another question. Please try again in a moment.", status=429, mimetype="text/plain")

    def stream():
        try:
            for chunk in state["llm"].create_chat_completion(
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": msg}],
                max_tokens=MAX_NEW_TOKENS, temperature=0.0, repeat_penalty=1.1, stream=True):
                piece = chunk["choices"][0]["delta"].get("content")
                if piece:
                    yield piece
        finally:
            gen_lock.release()

    return Response(stream_with_context(stream()), mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
