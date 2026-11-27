// PinkFlag frontend: talks to the FastAPI backend (/api/...). No framework, just the DOM.
const $ = (id) => document.getElementById(id);
const state = { guys: [], current: null, sending: false };

const STATUSES = ["talking stage", "situationship", "dating", "it's complicated", "ex"];
const PROFILE = [
  ["works_or_studies", "💼", "works / studies"],
  ["met_at", "📍", "where you met"],
  ["age", "🎂", "age"],
  ["detail", "✨", "a detail about him"],
];
const AVATAR = '<svg class="avatar"><use href="#sisi"/></svg>';

// ---------- helpers ----------
const escapeHtml = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// escape first, then allow **bold** from Sisi's replies
const formatReply = (s) => escapeHtml(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");

// ---------- hosted free version: no voice, hide those buttons ----------
fetch("/api/config").then((r) => r.json()).then((cfg) => {
  if (cfg.voice) return;
  for (const id of ["callBtn", "mic"]) document.getElementById(id)?.remove();
  const tag = document.querySelector(".who p");
  if (tag) tag.insertAdjacentHTML("beforeend",
    '<br><small>📞 voice calls run locally on GPU: see the demo video</small>');
}).catch(() => {});

// ---------- privacy notice (first visit) + "delete all my data" ----------
document.addEventListener("DOMContentLoaded", () => {
  const seen = (() => { try { return localStorage.getItem("pinkflag-privacy-ok"); } catch { return null; } })();
  if (!seen) document.getElementById("privacy").classList.add("open");
  document.getElementById("privacyOk").onclick = () => {
    try { localStorage.setItem("pinkflag-privacy-ok", "1"); } catch { /* private window */ }
    document.getElementById("privacy").classList.remove("open");
  };
  document.getElementById("deleteMe").onclick = async () => {
    if (!confirm("Delete every guy, chat and screenshot you added? This can't be undone.")) return;
    await fetch("/api/me", { method: "DELETE" });
    location.reload();
  };
});

// ---------- privacy: an anonymous random ID per browser, so visitors never see each other's data ----------
const VISITOR = (() => {
  try {
    let id = localStorage.getItem("pinkflag-visitor");
    if (!id) { id = crypto.randomUUID(); localStorage.setItem("pinkflag-visitor", id); }
    return id;
  } catch { return crypto.randomUUID(); }  // private window: a fresh ID per visit
})();
const _fetch = window.fetch.bind(window);
window.fetch = (url, opts = {}) => {
  if (String(url).startsWith("/api/")) {
    opts = { ...opts, headers: { ...(opts.headers || {}), "X-PinkFlag-User": VISITOR } };
  }
  return _fetch(url, opts);
};

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" }, ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok && res.status !== 503) throw new Error(data.detail || res.statusText);
  return data;
}

function toast(text) {
  const t = document.createElement("div");
  t.className = "toast"; t.textContent = text;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2600);
}

// ---------- guys / tabs ----------
async function loadGuys(selectId) {
  state.guys = await api("/api/guys");
  const hasGuys = state.guys.length > 0;
  $("empty").classList.toggle("hidden", hasGuys);
  $("main").classList.toggle("hidden", !hasGuys);
  if (!hasGuys) { state.current = null; renderTabs(); return; }
  const keep = selectId ?? state.current?.id;
  state.current = state.guys.find((g) => g.id === keep) || state.guys[0];
  renderTabs();
}

function renderTabs() {
  const tabs = $("tabs");
  tabs.innerHTML = "";
  for (const g of state.guys) {
    const b = document.createElement("button");
    b.textContent = `${g.emoji} ${g.name} · ${g.status}`;
    b.className = state.current && g.id === state.current.id ? "on" : "";
    b.onclick = () => selectGuy(g.id);
    tabs.appendChild(b);
  }
  const add = document.createElement("button");
  add.textContent = "+ new guy";
  add.onclick = addGuyPrompt;
  tabs.appendChild(add);
}

async function addGuyPrompt() {
  const name = prompt("His name (or a nickname)?");
  if (!name || !name.trim()) return;
  const guy = await api("/api/guys", { method: "POST", body: { name: name.trim() } });
  await loadGuys(guy.id);
  await selectGuy(guy.id);
}

async function selectGuy(id) {
  state.current = state.guys.find((g) => g.id === id);
  renderTabs();
  await Promise.all([loadChat(), loadReceipts()]);
}

// ---------- chat ----------
function addBubble(role, html, receipts = [], image = null) {
  const row = document.createElement("div");
  row.className = "row " + (role === "user" ? "me" : "sisi");
  for (const r of receipts) {
    if (r.confidence === null && Number.isInteger(r.id)) {
      $("thread").querySelectorAll(`[data-receipt-id="${r.id}"]`).forEach(card => card.remove());
    }
  }
  const flags = receipts.length
    ? `<div class="flags">${receipts.map((r) => `
        <div data-receipt-id="${Number(r.id)}" class="flag ${r.is_red_flag === null ? "pending" : r.is_red_flag ? "" : "good"}">
          <b>${r.emoji} ${escapeHtml(r.name)} · ${r.confidence === null ? "updated" : Math.round(r.confidence * 100) + "%"}</b>
          "${escapeHtml(r.text)}"
        </div>`).join("")}</div>`
    : "";
  row.innerHTML = (role === "user" ? "" : AVATAR) + `<div class="bubble">${html}${flags}</div>`;
  if (image) {
    const preview = document.createElement("img");
    preview.className = "chat-image";
    preview.alt = "Shared screenshot";
    preview.src = image instanceof Blob ? URL.createObjectURL(image) : image;
    if (image instanceof Blob) {
      preview.onload = preview.onerror = () => URL.revokeObjectURL(preview.src);
    }
    row.querySelector(".bubble").prepend(preview);
  }
  $("thread").appendChild(row);
  $("thread").scrollTop = $("thread").scrollHeight;
  return row;
}

async function loadChat() {
  setScreenshot(null);
  $("thread").innerHTML = "";
  const id = state.current.id;
  const history = await api(`/api/guys/${id}/chat`);
  if (id !== state.current?.id) return;
  if (!history.length) {
    addBubble("model", formatReply(
      `heyy bestie 💕 so, ${state.current.name}. tell me everything.\n` +
      `paste what he said (word for word is best) and I'll get you the receipts.`));
  }
  for (const m of history) {
    const image = m.image_id ? `/api/guys/${id}/images/${m.image_id}?u=${VISITOR}` : null;
    const text = image ? m.text.replace(/^\[Screenshot\]\s*/, "") : m.text;
    addBubble(m.role, m.role === "user" ? escapeHtml(text) : formatReply(text), [], image);
  }
}

let screenshot = null;
function setScreenshot(file) {
  if (file && (!["image/png", "image/jpeg"].includes(file.type) || file.size > 8 * 1024 * 1024)) {
    return toast("Choose a PNG or JPEG under 8 MB.");
  }
  const preview = $("draftImage");
  if (preview.hasAttribute("src")) URL.revokeObjectURL(preview.src);
  if (file) preview.src = URL.createObjectURL(file);
  else preview.removeAttribute("src");
  screenshot = file;
  $("attachment").hidden = !file;
  $("attachmentName").textContent = file?.name || "";
  $("imagePicker").value = "";
}
$("attachImage").onclick = () => $("imagePicker").click();
$("imagePicker").onchange = e => setScreenshot(e.target.files[0]);
$("removeImage").onclick = () => setScreenshot(null);
document.addEventListener("paste", e => {
  if (!state.current || !e.clipboardData || !$("composer").getClientRects().length) return;
  const imageItem = Array.from(e.clipboardData.items || []).find(
    item => item.kind === "file" && item.type.startsWith("image/")
  );
  const file = imageItem?.getAsFile() ||
    Array.from(e.clipboardData.files || []).find(f => f.type.startsWith("image/"));
  if (!file) return;
  e.preventDefault();
  setScreenshot(file);
  $("msg").focus();
});

async function send(text) {
  text = text.trim();
  const file = screenshot;
  if ((!text && !file) || state.sending || !state.current) return;
  const id = state.current.id;
  setScreenshot(null);
  state.sending = true; $("send").disabled = true;
  const sent = addBubble("user", escapeHtml(text), [], file);
  $("msg").value = ""; autosize();
  const typing = addBubble("model", "");
  typing.querySelector(".bubble").className = "bubble typing";
  typing.querySelector(".bubble").innerHTML = "<i></i><i></i><i></i>";
  try {
    let res;
    if (file) {
      const form = new FormData();
      form.append("image", file);
      form.append("text", text);
      const response = await fetch(`/api/guys/${id}/screenshot`, { method: "POST", body: form });
      res = await response.json();
      if (!response.ok) throw new Error(res.reply || res.detail || response.statusText);
    } else {
      res = await api(`/api/guys/${id}/chat`, { method: "POST", body: { text } });
    }
    if (state.current?.id !== id) { typing.remove(); return; }
    typing.remove();
    addBubble("model", formatReply(res.reply), res.receipts || []);
    await loadGuys(state.current.id);
    await loadReceipts();
  } catch (err) {
    typing.remove();
    if (state.current?.id !== id) return;
    if (file) sent.remove();
    if (file && !screenshot) {
      setScreenshot(file);
      if (!$("msg").value) $("msg").value = text;
    }
    addBubble("model", formatReply(`oops, something broke 😵 (${err.message})`));
  } finally {
    state.sending = false; $("send").disabled = false; $("msg").focus();
  }
}

function autosize() {
  const t = $("msg");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 140) + "px";
}

// ---------- receipts panel ----------
async function loadReceipts() {
  const data = await api(`/api/guys/${state.current.id}/receipts`);
  const g = data.guy;
  $("receiptsSub").textContent = `${g.emoji} ${g.name} · ${g.analysed} message(s) analysed - ${g.needs_context} need context`;
  $("pin").hidden = g.score == null;
  $("pin").style.left = (g.score ?? 50) + "%";
  $("verdict").textContent = g.verdict;
  $("verdict").style.color = g.score == null ? "var(--muted)"
    : g.score < 35 ? "var(--green)" : g.score < 65 ? "var(--amber)" : "var(--red)";
  $("statAnalysed").textContent = g.analysed;
  $("statRed").textContent = g.red_flags;
  $("statGreen").textContent = g.green_flags;

  const profile = $("profile");
  profile.innerHTML = "";
  const statusLi = document.createElement("li");
  statusLi.innerHTML = `<span>💌</span><span class="v"></span>`;
  const sel = document.createElement("select");
  sel.className = "status-select";
  for (const s of [...new Set([...STATUSES, g.status])]) {
    const o = document.createElement("option"); o.textContent = s; o.selected = s === g.status;
    sel.appendChild(o);
  }
  sel.onchange = async () => {
    await api(`/api/guys/${g.id}`, { method: "PATCH", body: { status: sel.value } });
    await loadGuys(g.id);
  };
  statusLi.querySelector(".v").appendChild(sel);
  profile.appendChild(statusLi);
  for (const [field, icon, label] of PROFILE) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${icon}</span><span class="v"></span><button class="edit" title="edit">✏️</button>`;
    li.querySelector(".v").textContent = g[field] || `(${label}: Sisi fills this in as you talk)`;
    if (!g[field]) li.querySelector(".v").classList.add("muted");
    li.querySelector(".edit").onclick = async () => {
      const value = prompt(`${label}?`, g[field] || "");
      if (value === null) return;
      await api(`/api/guys/${g.id}`, { method: "PATCH", body: { [field]: value } });
      await loadReceipts();
    };
    profile.appendChild(li);
  }

  const tl = $("timeline");
  tl.innerHTML = data.timeline.length ? "" : `<li class="muted">nothing analysed yet</li>`;
  for (const r of data.timeline.slice().reverse()) {
    const li = document.createElement("li");
    li.innerHTML = `<div></div><span class="chip ${r.is_red_flag === null ? "pending" : r.is_red_flag ? "" : "good"}"></span>`;
    li.querySelector("div").textContent = `"${r.text}"`;
    li.querySelector(".chip").textContent = `${r.emoji} ${r.name}`;
    const correct = document.createElement("select");
    correct.setAttribute("aria-label", "Correct rating for " + r.text);
    correct.add(new Option("Correct rating...", ""));
    for (const [value, name] of Object.entries(data.rating_options)) {
      correct.add(new Option(name, value));
    }
    correct.onchange = async () => {
      if (!correct.value) return;
      correct.disabled = true;
      try {
        await api(`/api/guys/${g.id}/receipts/${r.id}`, {
          method: "PATCH", body: {label: correct.value}
        });
        if (state.current?.id === g.id) {
          await loadGuys(g.id);
          await loadChat();
          await loadReceipts();
        }
      } catch (error) {
        toast(error.message);
        correct.disabled = false;
      }
    };
    li.appendChild(correct);
    tl.appendChild(li);
  }

  const notes = $("notes");
  notes.innerHTML = data.notes.length ? "" : `<li class="muted">nothing yet</li>`;
  for (const n of data.notes) {
    const li = document.createElement("li"); li.textContent = "📝 " + n; notes.appendChild(li);
  }
}

$("deleteGuy").onclick = async () => {
  if (!state.current) return;
  if (!confirm(`Delete ${state.current.name} and everything about him? This can't be undone.`)) return;
  await api(`/api/guys/${state.current.id}`, { method: "DELETE" });
  state.current = null;
  await loadGuys();
  if (state.current) await selectGuy(state.current.id);
};

// ---------- roster + wheel ----------
let rotation = 0;
const SLICE_COLORS = ["#F4B6C8", "#E88FAB", "#FBEFF3", "#F9D3DF", "#D98BA8", "#FCE4EC"];

function renderRoster() {
  const box = $("guys");
  box.innerHTML = "";
  for (const g of state.guys) {
    const card = document.createElement("div");
    card.className = "guy"; card.dataset.id = g.id;
    const prof = PROFILE.filter(([f]) => g[f]).map(([f, icon]) => `<li>${icon} <span data-f="${f}"></span></li>`).join("");
    card.innerHTML = `
      <div class="face">${escapeHtml(g.emoji)}</div>
      <h3></h3><span class="status"></span>
      ${prof ? `<ul class="profile">${prof}</ul>` : ""}
      <div class="mini"><i ${g.score == null ? "hidden" : ""} style="left:${g.score ?? 50}%"></i></div>
      <div class="muted">💍 ←→ 🚩 · <b class="v"></b></div>
      <div class="tops"></div>`;
    card.querySelector("h3").textContent = g.name;
    card.querySelector(".status").textContent = g.status;
    card.querySelectorAll("[data-f]").forEach((s) => (s.textContent = g[s.dataset.f]));
    card.querySelector(".v").textContent = g.verdict;
    for (const f of g.top_flags.slice(0, 3)) {
      const c = document.createElement("span");
      c.className = "chip" + (f.label === "green_flag" ? " good" : "");
      c.textContent = `${f.emoji} ${f.name} ×${f.count}`;
      card.querySelector(".tops").appendChild(c);
    }
    box.appendChild(card);
  }
  // wheel: one slice per guy
  const n = Math.max(state.guys.length, 1), slice = 360 / n;
  $("wheel").style.background = `conic-gradient(${state.guys.map((_, i) =>
    `${SLICE_COLORS[i % SLICE_COLORS.length]} ${i * slice}deg ${(i + 1) * slice}deg`).join(", ")})`;
  $("wheel").innerHTML = state.guys.map((g, i) =>
    `<span style="transform: rotate(${i * slice + slice / 2}deg) translateY(-32cqw) translateX(-50%)">${escapeHtml(g.emoji)} ${escapeHtml(g.name)}</span>`).join("");
}

$("openRoster").onclick = async () => {
  await loadGuys();
  renderRoster();
  $("reveal").classList.remove("show");
  $("roster").classList.add("open");
};
$("closeRoster").onclick = () => $("roster").classList.remove("open");
$("roster").onclick = (e) => { if (e.target === $("roster")) $("roster").classList.remove("open"); };

$("spin").onclick = async function () {
  if (state.guys.length < 2) return toast("add at least 2 guys to spin the wheel 🎡");
  let best;
  try { best = await api("/api/roster/best"); }
  catch { return toast("tell Sisi what they said first, no receipts yet 🧾"); }
  const idx = state.guys.findIndex((g) => g.id === best.id);
  const slice = 360 / state.guys.length;
  const centre = idx * slice + slice / 2;
  const wobble = (Math.random() - 0.5) * slice * 0.6;   // land inside the slice, not dead-centre
  const current = ((rotation % 360) + 360) % 360;
  rotation += 360 * 5 + ((360 - centre - wobble - current) % 360 + 360) % 360;
  this.disabled = true;
  document.querySelectorAll(".guy").forEach((c) => c.classList.remove("win"));
  $("reveal").classList.remove("show");
  $("wheel").style.transform = `rotate(${rotation}deg)`;
  $("wheel").addEventListener("transitionend", () => {
    document.querySelector(`.guy[data-id="${best.id}"]`)?.classList.add("win");
    const greens = best.green_flags, reds = best.red_flags;
    $("revealText").innerHTML = `<b></b><br><span></span>`;
    $("revealText").querySelector("b").textContent = `it's ${best.name} 💚`;
    $("revealText").querySelector("span").textContent =
      `${greens} green flag(s) vs ${reds} red flag(s) in what you shared: ${best.verdict}. ` +
      `the wheel picks the one who treats you best, but you always decide, bestie.`;
    $("reveal").classList.add("show");
    this.disabled = false;
  }, { once: true });
};

// ---------- shared mic recorder (used by dictation and calls) ----------
// Records with MediaRecorder and measures the loudness live, so you can SEE the mic hears you.
const mic = { stream: null, recorder: null, chunks: [], started: 0, peak: 0, raf: null };

async function startRecording(onLevel) {
  mic.stream = mic.stream || await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  const ctx = new AudioContext();
  const analyser = ctx.createAnalyser();
  ctx.createMediaStreamSource(mic.stream).connect(analyser);
  const data = new Uint8Array(analyser.fftSize);
  mic.peak = 0;
  const tick = () => {
    analyser.getByteTimeDomainData(data);
    let max = 0;
    for (const v of data) max = Math.max(max, Math.abs(v - 128));
    const level = Math.min(1, max / 64);
    mic.peak = Math.max(mic.peak, level);
    onLevel?.(level);
    mic.raf = requestAnimationFrame(tick);
  };
  tick();
  mic.chunks = [];
  mic.recorder = new MediaRecorder(mic.stream);
  mic.recorder.ondataavailable = (e) => mic.chunks.push(e.data);
  mic.recorder.start();
  mic.started = performance.now();
  mic.ctx = ctx;
}

function stopRecording() {
  return new Promise((resolve) => {
    mic.recorder.onstop = () => {
      cancelAnimationFrame(mic.raf);
      mic.ctx?.close();
      const ms = performance.now() - mic.started;
      resolve({ blob: new Blob(mic.chunks, { type: mic.recorder.mimeType }), ms, peak: mic.peak });
    };
    mic.recorder.stop();
  });
}
const isRecording = () => mic.recorder && mic.recorder.state === "recording";

// a recording is useless if it's too short or the mic heard (almost) nothing
function checkRecording(rec) {
  if (rec.ms < 700) return "that was super short 😅 tap, talk, then tap again";
  if (rec.peak < 0.04) return "I can't hear anything 🙉 check your mic (or Windows sound settings)";
  return null;
}

// ---------- 🎙️ dictation in the chat box: Whisper (local) writes what you say ----------
$("mic").onclick = async () => {
  if (isRecording()) {
    $("mic").classList.remove("on");
    $("msg").placeholder = "Sisi is writing down what you said… ✍️";
    const rec = await stopRecording();
    const problem = checkRecording(rec);
    if (problem) { $("msg").placeholder = "Tell Sisi what he said..."; return toast(problem); }
    const form = new FormData();
    form.append("audio", rec.blob, "voice.webm");
    try {
      const res = await (await fetch("/api/transcribe", { method: "POST", body: form })).json();
      if (!res.text) toast("didn't catch that, try again? 🎙️");
      $("msg").value = ($("msg").value ? $("msg").value + " " : "") + (res.text || "");
      autosize(); $("msg").focus();
    } catch { toast("transcription failed 😵"); }
    $("msg").placeholder = "Tell Sisi what he said...";
    return;
  }
  try { await startRecording((lvl) => ($("mic").style.boxShadow = `0 0 0 ${2 + lvl * 10}px rgba(215,38,61,.25)`)); }
  catch { return toast("I need mic permission to hear you 🎙️"); }
  $("mic").classList.add("on");
  $("msg").placeholder = "listening… tap 🎙️ again when you're done";
};

// ---------- wiring ----------
$("composer").onsubmit = (e) => { e.preventDefault(); send($("msg").value); };
$("msg").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send($("msg").value); }
});
$("msg").addEventListener("input", autosize);
document.querySelectorAll("#chips button").forEach((b) => (b.onclick = () => send(b.textContent)));
$("toggleReceipts").onclick = () => $("panel").classList.toggle("open");
// ---------- 📞 call Sisi: record -> /voice (Whisper -> Sisi -> Piper) -> play her answer ----------
const callState = (t) => ($("callState").textContent = t);

// 📳 ringtone made with the Web Audio API (two mixed tones, like a real phone): no sound file needed
let ringCtx = null, ringTimer = null;
function startRinging() {
  ringCtx = new (window.AudioContext || window.webkitAudioContext)();
  const ring = () => {
    const t = ringCtx.currentTime, gain = ringCtx.createGain();
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(0.15, t + 0.05);
    gain.gain.setValueAtTime(0.15, t + 0.9);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 1.0);
    gain.connect(ringCtx.destination);
    for (const f of [440, 480]) {
      const osc = ringCtx.createOscillator();
      osc.frequency.value = f; osc.connect(gain);
      osc.start(t); osc.stop(t + 1.0);
    }
  };
  ring();
  ringTimer = setInterval(ring, 2000);
}
function stopRinging() {
  clearInterval(ringTimer); ringTimer = null;
  ringCtx?.close(); ringCtx = null;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

$("callBtn").onclick = async () => {
  $("call").classList.add("open");
  $("talk").disabled = true;
  callState("calling Sisi… 📳");
  $("caption").textContent = "ringing…";
  startRinging();
  try {
    // ring at least ~2 times, and until her greeting is ready (the first one loads her voice model)
    const [greet] = await Promise.all([api("/api/voice/greeting"), sleep(3000)]);
    if (!$("call").classList.contains("open")) return;   // hung up while ringing
    stopRinging();
    callState("Sisi picked up 💕");
    $("caption").textContent = greet.text;
    $("sisiVoice").src = "data:audio/wav;base64," + greet.audio;
    $("sisiVoice").onended = () => callState("your turn, tap the mic 🎙️");
    await $("sisiVoice").play();
  } catch {
    stopRinging();
    callState("she couldn't pick up 😵 tap the mic to try anyway");
  } finally {
    $("talk").disabled = false;
  }
};
$("hang").onclick = async () => {
  stopRinging();
  if (isRecording()) await stopRecording();
  mic.stream?.getTracks().forEach((t) => t.stop()); mic.stream = null;
  $("sisiVoice").pause();
  $("talk").classList.remove("on");
  $("call").classList.remove("open");
};
$("talk").onclick = async () => {
  if (isRecording()) { await sendVoice(await stopRecording()); return; }
  $("sisiVoice").pause();
  try {
    await startRecording((lvl) => ($("talk").style.boxShadow = `0 0 0 ${4 + lvl * 22}px rgba(244,182,200,.45)`));
  } catch { return toast("I need your mic permission to hear you 🎙️"); }
  $("talk").classList.add("on");
  callState("listening… tap again when you're done");
};

async function sendVoice(rec) {
  $("talk").classList.remove("on");
  $("talk").style.boxShadow = "";
  const problem = checkRecording(rec);
  if (problem) { callState(problem); return; }
  $("talk").disabled = true;
  callState("Sisi is thinking… 💭");
  const form = new FormData();
  form.append("audio", rec.blob, "voice.webm");
  try {
    const res = await fetch(`/api/guys/${state.current.id}/voice`, { method: "POST", body: form });
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    if (data.transcript) addBubble("user", escapeHtml("🎙️ " + data.transcript));
    addBubble("model", formatReply(data.reply), data.receipts || []);
    $("caption").textContent = data.reply;
    callState("Sisi is talking 🔊");
    $("sisiVoice").src = "data:audio/wav;base64," + data.audio;
    $("sisiVoice").onended = () => callState("your turn, tap the mic 🎙️");
    await $("sisiVoice").play();
    loadGuys(state.current.id).then(loadReceipts);
  } catch (err) {
    callState("the line dropped 😵 try again?");
  } finally {
    $("talk").disabled = false;
  }
}
$("firstGuyForm").onsubmit = async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  const guy = await api("/api/guys", { method: "POST", body: { name: f.get("name"), status: f.get("status") } });
  await loadGuys(guy.id);
  await selectGuy(guy.id);
};

(async () => {
  await loadGuys();
  if (state.current) await selectGuy(state.current.id);
})();
