(function () {
  const OPTIONS = {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
    ],
    throwOnError: false,
    errorColor: "#cc0000",
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
  };

  // Merge any separated <p>$$</p> ... <p>$$</p> blocks into a single display math block
  function fixSplitMathParagraphs(root) {
    if (!root || !root.querySelectorAll) return;
    const paras = Array.from(root.querySelectorAll("p"));
    for (let i = 0; i < paras.length; i++) {
      const p = paras[i];
      const text = (p.textContent || "").trim();
      if (text === "$$" || text === "\\[") {
        const closingDelim = text === "$$" ? "$$" : "\\]";
        let j = i + 1;
        const middle = [];
        let found = false;
        while (j < paras.length) {
          const t2 = (paras[j].textContent || "").trim();
          if (t2 === closingDelim) {
            found = true;
            break;
          }
          middle.push(paras[j]);
          j++;
        }
        if (found) {
          const combined = middle.map(el => (el.textContent || "").trim()).join(" ");
          const div = document.createElement("div");
          div.className = "math-block";
          div.style.cssText = "margin:14px 0; text-align:center; overflow-x:auto;";
          div.textContent = text + " " + combined + " " + closingDelim;
          p.parentNode.insertBefore(div, p);
          p.remove();
          middle.forEach(el => el.remove());
          paras[j].remove();
          i = j;
        }
      }
    }
  }

  // Clean empty lines inside $$ ... $$ blocks in text nodes so KaTeX does not encounter illegal \par
  function cleanMathInDOM(root) {
    if (!root || typeof document === "undefined" || !document.createTreeWalker) return;
    try {
      const walker = document.createTreeWalker(
        root,
        NodeFilter.SHOW_TEXT,
        {
          acceptNode: function (node) {
            if (!node || !node.nodeValue) return NodeFilter.FILTER_REJECT;
            const parent = node.parentElement;
            if (parent && /^(script|style|textarea|code|pre|math-field)$/i.test(parent.tagName)) {
              return NodeFilter.FILTER_REJECT;
            }
            if (parent && parent.closest(".katex")) {
              return NodeFilter.FILTER_REJECT;
            }
            if (node.nodeValue.includes("$$") || node.nodeValue.includes("\\[")) {
              return NodeFilter.FILTER_ACCEPT;
            }
            return NodeFilter.FILTER_REJECT;
          }
        }
      );
      const textNodes = [];
      while (walker.nextNode()) textNodes.push(walker.currentNode);
      textNodes.forEach(node => {
        node.nodeValue = node.nodeValue
          .replace(/\$\$([\s\S]*?)\$\$/g, function (_, inner) {
            const clean = inner.replace(/^\s+|\s+$/g, "").replace(/\n\s*\n+/g, " ");
            return "$$" + clean + "$$";
          })
          .replace(/\\\[([\s\S]*?)\\\]/g, function (_, inner) {
            const clean = inner.replace(/^\s+|\s+$/g, "").replace(/\n\s*\n+/g, " ");
            return "\\[" + clean + "\\]";
          });
      });
    } catch (_) {}
  }

  function renderMath(root) {
    if (!window.renderMathInElement) return;
    const target = root || document.body;
    fixSplitMathParagraphs(target);
    cleanMathInDOM(target);
    renderMathInElement(target, OPTIONS);
  }
  window.renderMath = renderMath; // call after inserting content manually

  // Client mirror of the server's wrap_bare_latex().
  const LATEX_CMD = /\\[a-zA-Z]+/;
  const LATEX_RUN = /\\[a-zA-Z]+(?:\s*\{(?:[^{}]|\{[^{}]*\})*\}){0,2}/g;
  const DELIMITED = /(\$\$[\s\S]*?\$\$|\$[^$]*\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\))/g;

  const MATH_WORDS = /^(?:sin|cos|tan|cot|sec|csc|log|ln|lim|max|min|det|pi|theta|arcsin|arccos|arctan|sinh|cosh|tanh)$/i;

  function isMathLine(piece) {
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
      if (isMathLine(piece) && !piece.includes("$")) {
        const m = /^(\s*)([\s\S]*?)(\s*)$/.exec(piece);
        return m[1] + "$" + m[2] + "$" + m[3];
      }
      return piece.replace(LATEX_RUN, function (run) {
        return run.replace(/\s/g, "").length > 1 ? "$" + run + "$" : run;
      });
    }).join("");
  }
  window.wrapBareLatex = wrapBareLatex;

  // markdown-lite: parses headings, bold, code, lists, and preserves display math blocks intact
  function renderMarkdownLite(src) {
    if (src == null) return "";
    let h = String(src);

    // 1. Extract and preserve display math blocks ($$...$$ and \[...\])
    const mathBlocks = [];
    h = h.replace(/\$\$([\s\S]*?)\$\$/g, function (_, inner) {
      const clean = inner.replace(/^\s+|\s+$/g, "").replace(/\n\s*\n+/g, " ");
      const idx = mathBlocks.length;
      mathBlocks.push("$$" + clean + "$$");
      return "\n\n___MATH_BLOCK_" + idx + "___\n\n";
    });
    h = h.replace(/\\\[([\s\S]*?)\\\]/g, function (_, inner) {
      const clean = inner.replace(/^\s+|\s+$/g, "").replace(/\n\s*\n+/g, " ");
      const idx = mathBlocks.length;
      mathBlocks.push("\\[" + clean + "\\]");
      return "\n\n___MATH_BLOCK_" + idx + "___\n\n";
    });

    // 2. Escape HTML
    h = h.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    // 3. Inline markdown: bold and code
    h = h
      .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");

    // 4. Block formatting: lists, headings, math blocks, paragraphs
    let out = "";
    let inList = false;
    for (const line of h.split("\n")) {
      const t = line.trim();
      const mathMatch = t.match(/^___MATH_BLOCK_(\d+)___$/);
      const item = t.match(/^(?:\d+[.)]|[-*])\s+(.*)$/);
      const head = t.match(/^(#{1,4})\s+(.*)$/);

      if (mathMatch) {
        if (inList) { out += "</ol>"; inList = false; }
        const idx = parseInt(mathMatch[1], 10);
        out += '<div class="math-block" style="margin:14px 0; text-align:center; overflow-x:auto;">' + mathBlocks[idx] + "</div>";
      } else if (item) {
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
