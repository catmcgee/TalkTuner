/* TalkTuner dashboard frontend: chat with the model on the left, watch (and
   pin) its internal beliefs about you on the right. */

const ATTR_LABELS = {
  age: "Age",
  gender: "Gender",
  education: "Education",
  socioeco: "Socioeconomic status",
};

const CLASS_LABELS = {
  child: "Child", adolescent: "Teen", adult: "Adult", "older adult": "Older adult",
  female: "Female", male: "Male",
  someschool: "Some school", highschool: "High school", collegemore: "College+",
  low: "Lower income", middle: "Middle income", high: "Higher income",
};

const messages = [];   // {role, content}
const pins = {};       // attr -> class | null
let config = null;
let topClasses = {};   // attr -> current top class (for flip detection)
let busy = false;

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages"), cardsEl = $("cards"), inputEl = $("input");
const statusEl = $("mirror-status"), sendBtn = $("send");

init();

async function init() {
  config = await (await fetch("/api/config")).json();
  $("model-name").textContent = config.mode === "paper"
    ? `${config.model} — the paper's original probes`
    : `${config.model} — probes trained on its activations`;
  $("alpha").max = config.max_alpha;
  $("alpha").value = config.default_alpha;
  $("alpha-value").textContent = config.default_alpha;
  for (const [attr, info] of Object.entries(config.attributes)) {
    pins[attr] = null;
    cardsEl.appendChild(buildCard(attr, info));
  }
  $("alpha").addEventListener("input", (e) => {
    $("alpha-value").textContent = e.target.value;
  });
  $("composer").addEventListener("submit", (e) => { e.preventDefault(); send(); });
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  $("new-chat").addEventListener("click", resetChat);
}

function buildCard(attr, info) {
  const card = document.createElement("div");
  card.className = "card";
  card.id = `card-${attr}`;
  const acc = Math.round(info.val_acc * 100);
  card.innerHTML = `
    <div class="card-head">
      <span class="card-title">${ATTR_LABELS[attr] || attr}</span>
      <span class="card-meta">probe ${acc}% · layer ${info.layer}</span>
    </div>`;
  for (const cls of info.classes) {
    const row = document.createElement("div");
    row.className = "belief";
    row.dataset.attr = attr;
    row.dataset.cls = cls;
    row.innerHTML = `
      <span class="belief-label">${CLASS_LABELS[cls] || cls}</span>
      <div class="meter"><div class="meter-fill"></div></div>
      <span class="belief-pct">–</span>
      <button class="pin-btn" type="button"
        aria-label="Pin ${ATTR_LABELS[attr]} as ${CLASS_LABELS[cls] || cls}"
        title="Make the model believe this">⏚</button>`;
    row.querySelector(".pin-btn").addEventListener("click", () => togglePin(attr, cls));
    card.appendChild(row);
  }
  return card;
}

function togglePin(attr, cls) {
  pins[attr] = pins[attr] === cls ? null : cls;
  const card = $(`card-${attr}`);
  card.classList.toggle("pinned", !!pins[attr]);
  card.querySelectorAll(".belief").forEach((row) => {
    row.classList.toggle("pinned", row.dataset.cls === pins[attr]);
  });
  const meta = card.querySelector(".card-meta");
  if (pins[attr]) {
    meta.innerHTML = `<span class="pinned-tag">pinned: ${CLASS_LABELS[cls] || cls}</span>`;
  } else {
    const info = config.attributes[attr];
    meta.textContent = `probe ${Math.round(info.val_acc * 100)}% · layer ${info.layer}`;
  }
  const anyPin = Object.values(pins).some(Boolean);
  setStatus(anyPin ? "steering active — next replies are intervened" : "", anyPin);
}

function applyReadings(readings) {
  for (const [attr, probs] of Object.entries(readings)) {
    const top = Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0];
    const card = $(`card-${attr}`);
    if (topClasses[attr] && topClasses[attr] !== top) {
      card.classList.remove("flipped");
      void card.offsetWidth;  // restart the border-flash transition
      card.classList.add("flipped");
      setTimeout(() => card.classList.remove("flipped"), 1600);
    }
    topClasses[attr] = top;
    card.querySelectorAll(".belief").forEach((row) => {
      const p = probs[row.dataset.cls] ?? 0;
      row.classList.toggle("top", row.dataset.cls === top);
      row.querySelector(".meter-fill").style.width = `${(p * 100).toFixed(1)}%`;
      row.querySelector(".belief-pct").textContent = `${(p * 100).toFixed(0)}%`;
    });
  }
}

function setStatus(text, live = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("live", live);
}

function addBubble(role, text) {
  const empty = messagesEl.querySelector(".chat-empty");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

async function send() {
  const text = inputEl.value.trim();
  if (!text || busy) return;
  busy = true;
  sendBtn.disabled = true;
  inputEl.value = "";

  messages.push({ role: "user", content: text });
  addBubble("user", text);
  const bubble = addBubble("assistant", "");
  bubble.classList.add("thinking");
  setStatus("reading…", true);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, pins, alpha: Number($("alpha").value) }),
    });
    if (!res.ok) throw new Error(`server returned ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "", reply = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const events = buf.split("\n\n");
      buf = events.pop();
      for (const ev of events) {
        if (!ev.startsWith("data: ")) continue;
        const msg = JSON.parse(ev.slice(6));
        if (msg.type === "token") {
          reply += msg.text;
          bubble.textContent = reply;
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (msg.type === "readings") {
          applyReadings(msg.readings);
          setStatus("generating…", true);
        } else if (msg.type === "done") {
          reply = msg.reply;
          bubble.textContent = reply;
          applyReadings(msg.readings);
        }
      }
    }
    messages.push({ role: "assistant", content: reply });
    const anyPin = Object.values(pins).some(Boolean);
    setStatus(anyPin ? "steering active — next replies are intervened" : "", anyPin);
  } catch (err) {
    bubble.textContent = `Something went wrong: ${err.message}. ` +
      "Check that the server is still running, then try again.";
    setStatus("");
    messages.pop();  // let the user resend their message
  } finally {
    bubble.classList.remove("thinking");
    busy = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

function resetChat() {
  messages.length = 0;
  topClasses = {};
  messagesEl.innerHTML = `<div class="chat-empty"><p>Say anything. From your
    very first message, the model starts forming a picture of who you are —
    age, gender, education, income. The panel on the right shows that picture
    as it forms.</p></div>`;
  document.querySelectorAll(".meter-fill").forEach((el) => (el.style.width = "0"));
  document.querySelectorAll(".belief-pct").forEach((el) => (el.textContent = "–"));
  document.querySelectorAll(".belief.top").forEach((el) => el.classList.remove("top"));
  setStatus("No reading yet — send a message.");
}
