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
    renderMath(document.body);
    observer.observe(document.body, { childList: true, subtree: true });
  });
})();
