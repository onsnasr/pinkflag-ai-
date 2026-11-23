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
function addBubble(role, html, receipts = []) {
  const row = document.createElement("div");
  row.className = "row " + (role === "user" ? "me" : "sisi");
  const flags = receipts.length
    ? `<div class="flags">${receipts.map((r) => `
        <div class="flag ${r.is_red_flag ? "" : "good"}">
          <b>${r.emoji} ${escapeHtml(r.name)} · ${Math.round(r.confidence * 100)}%</b>
          "${escapeHtml(r.text)}"
        </div>`).join("")}</div>`
    : "";
  row.innerHTML = (role === "user" ? "" : AVATAR) + `<div class="bubble">${html}${flags}</div>`;
  $("thread").appendChild(row);
  $("thread").scrollTop = $("thread").scrollHeight;
  return row;
}

async function loadChat() {
  $("thread").innerHTML = "";
  const history = await api(`/api/guys/${state.current.id}/chat`);
  if (!history.length) {
    addBubble("model", formatReply(
      `heyy bestie 💕 so, ${state.current.name}. tell me everything.\n` +
      `paste what he said (word for word is best) and I'll get you the receipts.`));
  }
  for (const m of history) {
    addBubble(m.role, m.role === "user" ? escapeHtml(m.text) : formatReply(m.text));
  }
}

async function send(text) {
  text = text.trim();
  if (!text || state.sending || !state.current) return;
  state.sending = true; $("send").disabled = true;
  addBubble("user", escapeHtml(text));
  $("msg").value = ""; autosize();
  const typing = addBubble("model", "");
  typing.querySelector(".bubble").className = "bubble typing";
  typing.querySelector(".bubble").innerHTML = "<i></i><i></i><i></i>";
  try {
    const res = await api(`/api/guys/${state.current.id}/chat`, { method: "POST", body: { text } });
    typing.remove();
    addBubble("model", formatReply(res.reply), res.receipts || []);
    await loadGuys(state.current.id);
    await loadReceipts();
  } catch (err) {
    typing.remove();
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
  $("receiptsSub").textContent = `${g.emoji} ${g.name} · ${g.analysed} message(s) analysed`;
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
    li.innerHTML = `<div></div><span class="chip ${r.is_red_flag ? "" : "good"}"></span>`;
    li.querySelector("div").textContent = `"${r.text}"`;
    li.querySelector(".chip").textContent = `${r.emoji} ${r.name}`;
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
      <div class="mini"><i style="left:${g.score ?? 50}%"></i></div>
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
    `<span style="transform: rotate(${i * slice + slice / 2}deg) translateY(-96px) translateX(-50%)">${escapeHtml(g.emoji)} ${escapeHtml(g.name)}</span>`).join("");
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

// ---------- dictation (browser speech recognition) ----------
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null;
$("mic").onclick = () => {
  if (!SR) return toast("voice typing needs Chrome or Edge 🙏");
  if (rec) { rec.stop(); return; }
  rec = new SR();
  rec.lang = navigator.language || "en-US";
  rec.interimResults = true; rec.continuous = true;
  const before = $("msg").value ? $("msg").value + " " : "";
  rec.onresult = (e) => {
    let text = "";
    for (const r of e.results) text += r[0].transcript;
    $("msg").value = before + text; autosize();
  };
  rec.onend = () => { rec = null; $("mic").classList.remove("on"); $("msg").placeholder = "Tell Sisi what he said..."; };
  rec.onerror = () => toast("couldn't hear you, check the mic permission 🎙️");
  rec.start();
  $("mic").classList.add("on");
  $("msg").placeholder = "listening… tap 🎙️ again to stop";
};

// ---------- wiring ----------
$("composer").onsubmit = (e) => { e.preventDefault(); send($("msg").value); };
$("msg").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send($("msg").value); }
});
$("msg").addEventListener("input", autosize);
document.querySelectorAll("#chips button").forEach((b) => (b.onclick = () => send(b.textContent)));
$("toggleReceipts").onclick = () => $("panel").classList.toggle("open");
$("callBtn").onclick = () => $("call").classList.add("open");
$("hang").onclick = () => $("call").classList.remove("open");
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
