(function () {
  "use strict";

  // Floating Math Tutor: student-only help widget. Read-only w.r.t. page
  // state — it never modifies lessons, drafts, or submissions. Correctness
  // is decided server-side by SymPy (/api/tutor/message); the client only
  // renders reply types and forwards pageContext.
  var OPEN_KEY = "mathpilot-ft-open";
  var TIMEOUT_MS = 20000;

  function $(id) { return document.getElementById(id); }

  function pageContext() {
    var ctx = { type: "general" };
    try {
      var path = window.location.pathname || "";
      // Assignment workspace: graded vs practice from the page state.
      var qnav = $("q-nav");
      if (qnav || document.querySelector(".q-pane")) {
        var panes = document.querySelectorAll(".q-pane");
        var qi = 0;
        try { qi = qnav ? parseInt(qnav.value || "0", 10) : 0; } catch (e) { qi = 0; }
        var pane = panes[qi];
        var qtext = "";
        if (pane) {
          var p = pane.querySelector("p");
          qtext = p ? p.textContent.trim() : "";
        }
        var submitted = document.body.textContent.indexOf("Awaiting Review") !== -1;
        ctx = {
          type: "assignment",
          assignmentId: (typeof LINK_ID !== "undefined" && LINK_ID) || path,
          questionId: "q" + qi,
          questionText: qtext,
          mode: submitted ? "practice" : "graded"
        };
      } else if (path.indexOf("/learn/") === 0 || path.indexOf("/mylessons/") === 0) {
        var title = document.querySelector(".page-title");
        var sub = document.querySelector(".page-subtitle");
        var secs = [];
        document.querySelectorAll(".sec-preview").forEach(function (el) {
          secs.push(el.textContent.trim().slice(0, 400));
        });
        ctx = {
          type: "lesson",
          lessonId: path,
          sectionTitle: title ? title.textContent.trim() : "",
          sectionText: ((sub ? sub.textContent.trim() + " " : "") + secs.join(" ")).slice(0, 1200)
        };
      }
      if (typeof CURRENT_COURSE !== "undefined" && CURRENT_COURSE) ctx.courseId = CURRENT_COURSE;
      var picker = document.querySelector(".course-picker-title span");
      if (picker && picker.textContent.trim()) ctx.course = picker.textContent.trim();
    } catch (e) { /* context is advisory only */ }
    return ctx;
  }
  window.MathPilotPageContext = pageContext;

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function renderStepStrip(el, idx, total) {
    if (!el) return;
    if (!total) { el.textContent = ""; el.setAttribute("aria-label", "No active problem"); return; }
    var html = "";
    for (var i = 0; i < total; i++) {
      var cls = i < idx ? "done" : (i === idx ? "current" : "upcoming");
      html += '<span class="sdot ' + cls + '"></span>';
    }
    html += " Step " + Math.min(idx + 1, total) + " of " + total;
    el.innerHTML = html;
    el.setAttribute("aria-label", "Step " + Math.min(idx + 1, total) + " of " + total);
  }

  function addMsg(role, html, cls) {
    var body = $("float-tutor-body");
    if (!body) return null;
    var empty = body.querySelector(".float-tutor-empty");
    if (empty) empty.remove();
    var div = document.createElement("div");
    div.className = "float-tutor-msg " + (role === "student" ? "student" : "tutor");
    if (role === "student") {
      div.innerHTML = '<div class="bubble"></div>';
      div.firstChild.textContent = html;
    } else {
      div.innerHTML = '<span class="who" aria-hidden="true">&#129302;</span><div class="' + (cls || "bubble") + '">' + html + "</div>";
    }
    body.appendChild(div);
    body.scrollTop = body.scrollHeight;
    if (window.renderMath) window.renderMath(div);
    return div;
  }

  function showTyping() {
    var el = addMsg("tutor", "<span></span><span></span><span></span>", "bubble float-tutor-typing");
    if (el) el.id = "float-tutor-typing";
    return el;
  }
  function hideTyping() {
    var el = $("float-tutor-typing");
    if (el) el.remove();
  }

  function showRecap(recap) {
    var body = $("float-tutor-body");
    if (!body || !recap) return;
    var div = document.createElement("div");
    div.className = "float-tutor-msg tutor";
    var steps = "";
    (recap.steps || []).forEach(function (s) {
      steps += "<li><b>Step " + s.n + ":</b> " + esc(s.action) +
        "<br><span>You wrote: $" + esc(s.own || s.result) + "$</span><br><small>" +
        esc(s.why) + "</small></li>";
    });
    div.innerHTML = '<span class="who" aria-hidden="true">&#129302;</span>' +
      '<div class="float-tutor-recap"><div class="float-tutor-recap-head">' + esc(recap.title) + "</div>" +
      '<div class="float-tutor-recap-body"><ol>' + steps + "</ol>" +
      "<p><b>Answer: $" + esc(recap.answer) + "$</b></p>" +
      "<p>Check: $" + esc(recap.check) + "$</p>" +
      "<p><i>Key idea:</i> " + esc(recap.idea || "") + "</p>" +
      (recap.note ? "<p>" + esc(recap.note) + "</p>" : "") +
      '<div class="float-tutor-recap-actions">' +
      '<button type="button" data-act="similar">Try a similar problem</button>' +
      '<button type="button" data-act="new">Next question</button>' +
      '<button type="button" data-act="copy">Copy solution</button>' +
      "</div></div></div>";
    body.appendChild(div);
    body.scrollTop = body.scrollHeight;
    if (window.renderMath) window.renderMath(div);
    div.querySelectorAll("button").forEach(function (b) {
      b.addEventListener("click", function () {
        var act = b.getAttribute("data-act");
        if (act === "copy") {
          var txt = (recap.steps || []).map(function (s) {
            return "Step " + s.n + ": " + s.action + " -> " + s.result;
          }).join("\n") + "\nAnswer: " + recap.answer;
          try { navigator.clipboard.writeText(txt); } catch (e) { /* clipboard unavailable */ }
          addMsg("tutor", "Copied to your clipboard.");
        } else if (act === "similar") {
          send("Try a similar problem");
        } else {
          send("New problem");
        }
      });
    });
  }

  function setChips(names) {
    var box = $("float-tutor-quick");
    if (!box) return;
    box.innerHTML = "";
    var ctx = pageContext();
    var chips = names || [];
    if (!names) {
      chips = hasActive ? ["I'm stuck", "Explain this step", "New problem"]
        : ["10 = 3x + 1", "2x - 5 = 9", "I'm stuck", "New problem"];
      if (ctx.type === "assignment") chips.unshift("Help with this question");
      if (ctx.type === "lesson") chips.unshift("Explain this section");
    }
    chips.forEach(function (c) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "float-tutor-chip";
      b.textContent = c;
      b.addEventListener("click", function () { send(c); });
      box.appendChild(b);
    });
  }

  var hasActive = false;
  var busy = false;
  var lastText = "";

  function setBusy(on) {
    busy = on;
    var btn = $("float-tutor-send");
    var ta = $("float-tutor-text");
    if (btn) btn.disabled = on;
    if (ta) ta.disabled = on;
  }

  function markUnread() {
    var panel = $("float-tutor-panel");
    if (panel && !panel.classList.contains("open")) {
      var dot = $("float-tutor-unread");
      if (dot) dot.classList.add("show");
    }
  }

  function updateAfter(data) {
    renderStepStrip($("float-tutor-stepstrip"), data.stepIndex || 0, data.totalSteps || 0);
    var graded = $("float-tutor-graded");
    var ctx = pageContext();
    if (graded) graded.hidden = !(ctx.type === "assignment" && ctx.mode === "graded");
    hasActive = !!(data.totalSteps && !data.sessionDone);
    setChips();
    armIdle();
  }

  async function send(text) {
    text = (text || "").trim();
    if (!text || busy) return;
    clearTimeout(idleTimer);
    lastText = text;
    hideRetry();
    hideConfirm();
    addMsg("student", text);
    var ta = $("float-tutor-text");
    if (ta) { ta.value = ""; updateCount(); }
    setBusy(true);
    showTyping();
    var ctrl = new AbortController();
    var timer = setTimeout(function () { ctrl.abort(); }, TIMEOUT_MS);
    try {
      var r = await fetch("/api/tutor/message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text.slice(0, 500), pageContext: pageContext() }),
        signal: ctrl.signal
      });
      var j = await r.json().catch(function () { return {}; });
      hideTyping();
      if (!r.ok) {
        if (r.status === 429) addMsg("tutor", "Slow down a little — let's take this one step at a time. Try again in a bit.");
        else addMsg("tutor", "Sorry — " + esc((j && j.detail) || "that failed. Try again."));
        showRetry();
        setBusy(false);
        return;
      }
      var rtype = j.replyType || "bubble";
      if (rtype === "step") {
        var m = /Step\s+(\d+)\s*:\s*(.*)/i.exec(j.reply || "");
        if (m) addMsg("tutor", "<strong>Step " + m[1] + "</strong><br>" + esc(m[2]), "float-tutor-stepcard");
        else addMsg("tutor", esc(j.reply || ""));
      } else if (rtype === "hint") {
        var hm = /Hint\s+(\d+)\s*[—\-–:]\s*(.*)/i.exec(j.reply || "");
        // The ladder has four rungs; the student sees three hints and then a
        // split-into-smaller-steps card, which is what "of 3" promises.
        var label = !hm ? "Hint" : (Number(hm[1]) >= 4
          ? "Smaller steps" : "Hint " + hm[1] + " of 3");
        if (hm) addMsg("tutor", "&#128161; <strong>" + label + "</strong><br>" +
          esc(hm[2]), "float-tutor-hintcard");
        else addMsg("tutor", "&#128161; " + esc(j.reply || ""), "float-tutor-hintcard");
      } else if (rtype === "support") {
        addMsg("tutor", esc(j.reply || ""), "bubble float-tutor-support");
      } else if (rtype === "success") {
        addMsg("tutor", "&#10003; " + esc(j.reply || ""), "bubble float-tutor-success");
      } else if (rtype === "recap") {
        addMsg("tutor", esc(j.reply || ""));
        showRecap(j.recap);
      } else if (rtype === "confirm_new_problem") {
        addMsg("tutor", esc(j.reply || "That looks like a new problem. Start it now?"));
        showConfirm();
      } else {
        addMsg("tutor", esc(j.reply || "(no reply)"));
      }
      if (j.verdict === "correct" || (j.reply || "").indexOf("Correct") === 0) {
        // success styling handled by step cards; keep it simple
      }
      markUnread();
      updateAfter(j);
    } catch (e) {
      hideTyping();
      addMsg("tutor", "That took too long — tap retry to send it again. Your text is saved below.");
      if (ta) { ta.value = lastText; updateCount(); }
      showRetry();
    }
    clearTimeout(timer);
    setBusy(false);
  }

  function showRetry() {
    var b = $("float-tutor-retry");
    if (b) b.hidden = false;
  }

  // Idle nudge: after ~90s of silence on an open step, the tutor offers one
  // gentle prompt. The server refuses a second nudge for the same step.
  var IDLE_MS = 90000;
  var idleTimer = null;

  async function nudge() {
    idleTimer = null;
    if (busy || !hasActive) return;
    try {
      var r = await fetch("/api/tutor/nudge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pageContext: pageContext() })
      });
      if (!r.ok) return;
      var j = await r.json();
      if (!j || !j.reply) return;
      addMsg("tutor", esc(j.reply), "bubble float-tutor-support");
      markUnread();
      updateAfter(j);
    } catch (e) { /* silence is acceptable; never spam */ }
  }

  function armIdle() {
    clearTimeout(idleTimer);
    if (!hasActive) return;
    idleTimer = setTimeout(nudge, IDLE_MS);
  }
  function hideRetry() {
    var b = $("float-tutor-retry");
    if (b) b.hidden = true;
  }

  function updateCount() {
    var ta = $("float-tutor-text");
    var c = $("float-tutor-count");
    if (!ta || !c) return;
    var n = ta.value.length;
    c.textContent = n >= 400 ? n + "/500" : "";
    autoGrow(ta);
    updatePreview(ta.value);
    armIdle();
  }

  function autoGrow(ta) {
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, parseFloat(getComputedStyle(ta).maxHeight) || 96) + "px";
  }

  function cleanPreview(src) {
    // Cheap client-side mirror of display_math (server is authoritative):
    // 2*x -> 2x, ** -> ^, pi -> π. Anything unparseable shows "...".
    var t = String(src || "");
    if (!/[0-9a-zA-Zπθ∞√^+\-*/()=<>!|]/.test(t)) return "";
    t = t.replace(/(\d)\s*\*\s*(?=[a-zA-Z(])/g, "$1");
    t = t.replace(/\*\s*(?=[a-zA-Z(])/g, "");
    t = t.replace(/\*\*(\d+)/g, "^$1");
    t = t.replace(/\bpi\b/g, "π").replace(/\btheta\b/g, "θ");
    return t.trim();
  }

  function updatePreview(val) {
    var p = $("float-tutor-preview");
    if (!p) return;
    var hasMath = /[0-9][+\-*/^()=]|[+\-*/^()=][0-9]|pi|sqrt|sin|cos|tan|log|lim|d\/dx|x\^|√|π|θ|∞/.test(val || "");
    if (!val || !hasMath) { p.hidden = true; p.textContent = ""; return; }
    var clean = cleanPreview(val);
    if (!clean) { p.hidden = true; p.textContent = ""; return; }
    p.hidden = false;
    p.textContent = clean;
    if (window.renderMath) { try { window.renderMath(p); } catch (e) { /* plain text fallback */ } }
  }

  function showConfirm() {
    var box = $("float-tutor-confirm");
    if (box) box.hidden = false;
  }
  function hideConfirm() {
    var box = $("float-tutor-confirm");
    if (box) box.hidden = true;
  }

  function emptyState() {
    var body = $("float-tutor-body");
    if (!body || body.children.length) return;
    var first = "there";
    try {
      var prof = document.querySelector(".user-name");
      if (prof && prof.textContent.trim()) first = prof.textContent.trim().split(/\s+/)[0];
    } catch (e) { /* keep default */ }
    var div = document.createElement("div");
    div.className = "float-tutor-empty";
    div.innerHTML = '<span class="big" aria-hidden="true">&#129302;</span>Hi ' + esc(first) +
      "! Send me an equation and I'll guide you step by step. I won't give you the answer, you'll find it yourself.";
    body.appendChild(div);
  }

  function toggle(force) {
    var panel = $("float-tutor-panel");
    var btn = $("float-tutor-btn");
    if (!panel || !btn) return;
    var open = typeof force === "boolean" ? force : !panel.classList.contains("open");
    panel.classList.toggle("open", open);
    panel.setAttribute("aria-hidden", open ? "false" : "true");
    btn.setAttribute("aria-expanded", open ? "true" : "false");
    var label = btn.querySelector(".float-tutor-label");
    if (label) label.textContent = open ? "Close" : "Ask Tutor";
    if (open) {
      var dot = $("float-tutor-unread");
      if (dot) dot.classList.remove("show");
      var ta = $("float-tutor-text");
      if (ta) ta.focus();
      try { sessionStorage.setItem(OPEN_KEY, "1"); } catch (e) { /* ignore */ }
    } else {
      try { sessionStorage.removeItem(OPEN_KEY); } catch (e) { /* ignore */ }
      btn.focus();
    }
  }

  async function resume() {
    try {
      var course = (pageContext().courseId) || "";
      var r = await fetch("/api/tutor/session?course=" + encodeURIComponent(course));
      if (!r.ok) return;
      var j = await r.json();
      if (j && j.active) {
        hasActive = true;
        renderStepStrip($("float-tutor-stepstrip"), j.stepIndex || 0, j.totalSteps || 0);
        addMsg("tutor", "Welcome back — resuming where you left off.<br>" + esc(j.prompt || ""));
        updateAfter({ stepIndex: j.stepIndex, totalSteps: j.totalSteps, sessionDone: false });
      }
    } catch (e) { /* resume is best-effort */ }
  }

  function init() {
    if (!$("float-tutor-panel")) return;
    emptyState();
    setChips();
    resume();
    $("float-tutor-btn").addEventListener("click", function () { toggle(); });
    $("float-tutor-close").addEventListener("click", function () { toggle(false); });
    $("float-tutor-min").addEventListener("click", function () { toggle(false); });
    $("float-tutor-new").addEventListener("click", function () { send("New problem"); });
    $("float-tutor-send").addEventListener("click", function () { send($("float-tutor-text").value); });
    $("float-tutor-retry").addEventListener("click", function () { send(lastText); });
    var ta = $("float-tutor-text");
    ta.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(ta.value); }
    });
    ta.addEventListener("input", updateCount);
    var fx = $("float-tutor-fx");
    // Default: toolbar open on desktop, closed on mobile.
    var tb0 = $("float-tutor-mathkeys");
    var mobile0 = window.matchMedia && window.matchMedia("(max-width: 640px)").matches;
    if (tb0) tb0.hidden = !!mobile0;
    if (fx) fx.setAttribute("aria-expanded", mobile0 ? "false" : "true");
    if (fx) fx.addEventListener("click", function () {
      var k = $("float-tutor-mathkeys");
      if (!k) return;
      k.hidden = !k.hidden;
      fx.setAttribute("aria-expanded", k.hidden ? "false" : "true");
    });
    var tools = $("float-tutor-mathkeys");
    if (tools) tools.querySelectorAll("button").forEach(function (b) {
      // Keep focus in the field (mobile keyboard stays open).
      b.addEventListener("pointerdown", function (e) { e.preventDefault(); });
      b.addEventListener("click", function () {
        var ins = b.getAttribute("data-ins") || "";
        var start = (ta.selectionStart == null ? ta.value.length : ta.selectionStart);
        var end = (ta.selectionEnd == null ? start : ta.selectionEnd);
        var sel = ta.value.slice(start, end);
        var caret = start + ins.length;
        if (sel && (ins === "()" || ins === "sqrt()" || ins === "sin()" ||
                    ins === "cos()" || ins === "tan()" || ins === "ln()" ||
                    ins === "log()" || ins === "^()")) {
          // Wrap selected text: sqrt + "x+1" -> sqrt(x+1).
          var open = ins.slice(0, ins.length - 1);
          ta.value = (ta.value.slice(0, start) + open + sel + ")" + ta.value.slice(end)).slice(0, 500);
          caret = start + open.length + sel.length + 1;
        } else if (ins === "||") {
          // Absolute value: || -> |cursor|.
          ta.value = (ta.value.slice(0, start) + "||" + ta.value.slice(end)).slice(0, 500);
          caret = start + 1;
        } else if (ins.slice(-1) === "(" && ins.length > 1) {
          // fn( or ^( : cursor goes inside.
          ta.value = (ta.value.slice(0, start) + ins + ta.value.slice(end)).slice(0, 500);
          caret = start + ins.length;
        } else if (ins === "()") {
          ta.value = (ta.value.slice(0, start) + "()" + ta.value.slice(end)).slice(0, 500);
          caret = start + 1;
        } else {
          ta.value = (ta.value.slice(0, start) + ins + ta.value.slice(end)).slice(0, 500);
          caret = start + ins.length;
        }
        ta.focus();
        try { ta.setSelectionRange(caret, caret); } catch (e) { /* ignore */ }
        updateCount();
      });
    });
    var confirm = $("float-tutor-confirm");
    if (confirm) confirm.querySelectorAll("button").forEach(function (b) {
      b.addEventListener("click", function () {
        hideConfirm();
        send(b.getAttribute("data-confirm") === "yes" ? "yes" : "no");
      });
    });
    // visualViewport: keep the panel above the phone keyboard.
    try {
      var vv = window.visualViewport;
      if (vv) vv.addEventListener("resize", function () {
        var panel = $("float-tutor-panel");
        if (panel && panel.classList.contains("open")) {
          panel.style.bottom = "0px";
          var body = $("float-tutor-body");
          if (body) body.scrollTop = body.scrollHeight;
        }
      });
    } catch (e) { /* visualViewport unavailable */ }
    // Hide the floating Ask-Tutor button while the panel is open on mobile.
    try {
      if (window.matchMedia && window.matchMedia("(max-width: 640px)").matches) {
        var btn = $("float-tutor-btn"), panel = $("float-tutor-panel");
        if (btn && panel) {
          new MutationObserver(function () {
            btn.style.display = panel.classList.contains("open") ? "none" : "";
          }).observe(panel, { attributes: true, attributeFilter: ["class"] });
        }
      }
    } catch (e) { /* MutationObserver unavailable */ }
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") toggle(false);
    });
    try {
      if (sessionStorage.getItem(OPEN_KEY) === "1") toggle(true);
    } catch (e) { /* ignore */ }
  }

  window.FloatTutor = { send: send, toggle: toggle, pageContext: pageContext };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
