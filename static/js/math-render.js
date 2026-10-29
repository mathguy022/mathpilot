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
