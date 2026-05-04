function tradelearnMermaidTheme() {
  return document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "default";
}

function tradelearnRenderMermaid() {
  if (typeof mermaid === "undefined") {
    return;
  }

  mermaid.initialize({
    startOnLoad: false,
    theme: tradelearnMermaidTheme(),
  });

  document.querySelectorAll("pre.mermaid").forEach(function (block) {
    var code = block.querySelector("code");
    if (code) {
      block.textContent = code.textContent;
    }
  });

  mermaid.run({
    nodes: document.querySelectorAll(".mermaid"),
  });
}

document$.subscribe(tradelearnRenderMermaid);
