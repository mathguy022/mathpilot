(function () {
  const OPTIONS = {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
    ],
    throwOnError: false,
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
  };

  function renderMath(root) {
    if (!window.renderMathInElement) return;
    renderMathInElement(root || document.body, OPTIONS);
  }
  window.renderMath = renderMath; // call after inserting content manually

  // Client mirror of the server's wrap_bare_latex(). The server only touches
  // its own replies, so a student's message — "g(x) = \frac{x^2 - 9}{x - 3}"
  // typed into the chat — reached the bubble as literal backslash code.
  const LATEX_CMD = /\\[a-zA-Z]+/;
  const LATEX_RUN = /\\[a-zA-Z]+(?:\s*\{(?:[^{}]|\{[^{}]*\})*\}){0,2}/g;
  const DELIMITED = /(\$\$[\s\S]*?\$\$|\$[^$]*\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\))/g;

  const MATH_WORDS = /^(?:sin|cos|tan|cot|sec|csc|log|ln|lim|max|min|det|pi|theta|arcsin|arccos|arctan|sinh|cosh|tanh)$/i;

  function isMathLine(piece) {
    // A line of working rather than a sentence about it: strip the commands
    // and the operators, and call it math only if no prose word survives.
    // Short tokens ("2t", "x2") are variables, not English.
    const prose = piece.replace(/\\[a-zA-Z]+/g, " ")
      .replace(/[{}()\[\]=+\-*/^.,;:<>!|_\\'"]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 2 && !MATH_WORDS.test(w));
    return prose.length === 0;
  }

  function wrapBareLatex(text) {
    const s = String(text == null ? "" : text);
    if (!LATEX_CMD.test(s)) return s;
    return s.split(DELIMITED).map(function (piece, i) {
      if (i % 2 || !LATEX_CMD.test(piece)) return piece; // already math
      // A stray unbalanced $ would break the span, so only take the whole line
      // when the dollar signs all sit inside already-delimited pieces.
      if (isMathLine(piece) && !piece.includes("$")) {
        const m = /^(\s*)([\s\S]*?)(\s*)$/.exec(piece);
        return m[1] + "$" + m[2] + "$" + m[3];
      }
      // Prose around the math: delimit each command run on its own.
      return piece.replace(LATEX_RUN, function (run) {
        return run.replace(/\s/g, "").length > 1 ? "$" + run + "$" : run;
      });
    }).join("");
  }
  window.wrapBareLatex = wrapBareLatex;

  // markdown-lite: the minimal subset lesson bodies actually use. Headings,
  // **bold**, `code`, and "- " / "1. " lists become real HTML; $...$ spans are
  // left alone so KaTeX can pick them up afterwards.
  function renderMarkdownLite(src) {
    let h = String(src == null ? "" : src)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    h = h
      .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
    let out = "";
    let inList = false;
    for (const line of h.split("\n")) {
      const t = line.trim();
      const item = t.match(/^(?:\d+[.)]|[-*])\s+(.*)$/);
      const head = t.match(/^(#{1,4})\s+(.*)$/);
      if (item) {
        if (!inList) { out += "<ol>"; inList = true; }
        out += "<li>" + item[1] + "</li>";
      } else {
        if (inList) { out += "</ol>"; inList = false; }
        if (head) out += "<h4>" + head[2] + "</h4>";
        else if (t) out += "<p>" + t + "</p>";
      }
    }
    if (inList) out += "</ol>";
    return out;
  }
  window.renderMarkdownLite = renderMarkdownLite;

  // Re-render automatically when content is added dynamically (chat, streaming, AJAX)
  let timer = null;
  const pending = new Set();
  const observer = new MutationObserver((mutations) => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        const el = node.nodeType === 1 ? node : node.parentElement;
        if (el && !el.closest(".katex")) pending.add(el);
      }
    }
    if (!pending.size) return;
    clearTimeout(timer);
    timer = setTimeout(() => {
      const batch = Array.from(pending);
      pending.clear();
      batch.forEach((el) => document.contains(el) && renderMath(el));
    }, 120);
  });

  window.addEventListener("load", () => {
    document.querySelectorAll("[data-md]").forEach((el) => {
      el.innerHTML = renderMarkdownLite(el.getAttribute("data-md"));
    });
    renderMath(document.body);
    observer.observe(document.body, { childList: true, subtree: true });
  });
})();
