const fs = require("node:fs");
const path = require("node:path");
const { marked } = require("marked");

const [inputArg, outputArg] = process.argv.slice(2);
if (!inputArg || !outputArg) {
  console.error("Usage: node render_manuscript_html.cjs input.md output.html");
  process.exit(2);
}

const inputPath = path.resolve(inputArg);
const outputPath = path.resolve(outputArg);
let markdown = fs.readFileSync(inputPath, "utf8");

const displayMath = [];
const inlineMath = [];

markdown = markdown.replace(/\$\$([\s\S]*?)\$\$/g, (_match, tex) => {
  const token = `MATHDISPLAYTOKEN${displayMath.length}X`;
  displayMath.push({ token, tex: tex.trim() });
  return `\n\n${token}\n\n`;
});

markdown = markdown.replace(/(?<!\\)\$([^$\n]+?)\$/g, (_match, tex) => {
  const token = `MATHINLINETOKEN${inlineMath.length}X`;
  inlineMath.push({ token, tex: tex.trim() });
  return token;
});

marked.setOptions({ gfm: true, breaks: false });
let body = marked.parse(markdown);

for (const { token, tex } of displayMath) {
  body = body.replace(
    `<p>${token}</p>`,
    `<div class="display-math">\\[${tex}\\]</div>`,
  );
}

for (const { token, tex } of inlineMath) {
  body = body.replaceAll(token, `<span class="inline-math">\\(${tex}\\)</span>`);
}

const titleMatch = markdown.match(/^#\s+(.+)$/m);
const title = titleMatch ? titleMatch[1] : path.basename(inputPath);

const html = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${title}</title>
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [["\\\\(", "\\\\)"]],
        displayMath: [["\\\\[", "\\\\]"]],
        tags: "ams"
      },
      chtml: { scale: 0.96 }
    };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    :root { color-scheme: light; }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: #171717;
      background: #f4f5f7;
      font-family: "Times New Roman", "Noto Serif SC", "Songti SC", SimSun, serif;
      font-size: 17px;
      line-height: 1.78;
    }
    main {
      width: min(1120px, calc(100% - 48px));
      margin: 28px auto 72px;
      padding: 58px 74px 80px;
      background: #fff;
      box-shadow: 0 2px 18px rgba(0, 0, 0, 0.08);
    }
    h1, h2, h3, h4 { color: #111; line-height: 1.3; }
    h1 { margin: 0 0 24px; text-align: center; font-size: 30px; }
    h2 { margin-top: 42px; padding-bottom: 7px; border-bottom: 1px solid #bbb; font-size: 25px; }
    h3 { margin-top: 31px; font-size: 21px; }
    h4 { margin-top: 25px; font-size: 18px; }
    p { margin: 0.75em 0; text-align: justify; }
    li { margin: 0.28em 0; }
    hr { margin: 34px 0; border: 0; border-top: 1px solid #bbb; }
    img { display: block; max-width: 100%; height: auto; margin: 22px auto; }
    table {
      display: block;
      width: max-content;
      max-width: 100%;
      margin: 20px auto;
      overflow-x: auto;
      border-collapse: collapse;
      font-size: 14px;
      line-height: 1.45;
    }
    th, td { padding: 7px 10px; border: 1px solid #aaa; }
    th { background: #f1f2f4; font-weight: 700; }
    code {
      padding: 0.08em 0.3em;
      border-radius: 3px;
      background: #f1f2f4;
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.9em;
    }
    .display-math {
      margin: 1.2em 0;
      padding: 0.25em 0;
      overflow-x: auto;
      text-align: center;
    }
    .inline-math { white-space: nowrap; }
    mjx-container[jax="CHTML"][display="true"] { margin: 0.8em 0 !important; }
    @media (max-width: 760px) {
      body { font-size: 16px; }
      main { width: 100%; margin: 0; padding: 30px 20px 54px; box-shadow: none; }
      h1 { font-size: 25px; }
      h2 { font-size: 22px; }
    }
    @media print {
      body { background: #fff; font-size: 11pt; }
      main { width: auto; margin: 0; padding: 0; box-shadow: none; }
      table { font-size: 8.5pt; }
    }
  </style>
</head>
<body>
  <main>${body}</main>
</body>
</html>
`;

fs.writeFileSync(outputPath, html, "utf8");
console.log(`Rendered ${inputPath} -> ${outputPath}`);
console.log(`Math blocks: ${displayMath.length}, inline expressions: ${inlineMath.length}`);
