import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { marked } = require("marked");

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function protectRichBlocks(markdown) {
  const blocks = [];
  const inline = [];

  const blockToken = (html) => {
    const token = `@@MANUSCRIPT_BLOCK_${blocks.length}@@`;
    blocks.push([token, html]);
    return `\n${token}\n`;
  };

  const inlineToken = (html) => {
    const token = `@@MANUSCRIPT_INLINE_${inline.length}@@`;
    inline.push([token, html]);
    return token;
  };

  let source = markdown.replace(
    /```([^\r\n]*)\r?\n([\s\S]*?)```/g,
    (_, language, code) =>
      blockToken(
        `<pre><code${language.trim() ? ` class="language-${escapeHtml(language.trim())}"` : ""}>${escapeHtml(code.replace(/\r?\n$/, ""))}</code></pre>`,
      ),
  );

  source = source.replace(/`([^`\r\n]+)`/g, (_, code) =>
    inlineToken(`<code>${escapeHtml(code)}</code>`),
  );

  source = source.replace(/\$\$\s*([\s\S]*?)\s*\$\$/g, (_, tex) =>
    blockToken(`<div class="display-math">\\[\n${tex.trim()}\n\\]</div>`),
  );

  source = source.replace(
    /(?<!\\)\$([^$\r\n]+?)(?<!\\)\$/g,
    (_, tex) => inlineToken(`<span class="inline-math">\\(${tex.trim()}\\)</span>`),
  );

  return { source, blocks, inline };
}

function renderMarkdown(markdown) {
  const protectedSource = protectRichBlocks(markdown);
  let html = marked.parse(protectedSource.source, { gfm: true });

  for (const [token, block] of protectedSource.blocks) {
    html = html
      .replace(`<p>${token}</p>\n`, `${block}\n`)
      .replace(token, block);
  }
  for (const [token, item] of protectedSource.inline) {
    html = html.replaceAll(token, item);
  }
  return html;
}

const [inputPath, templatePath, outputPath] = process.argv.slice(2);
if (!inputPath || !templatePath || !outputPath) {
  throw new Error(
    "Usage: node render_manuscript_html.mjs INPUT.md TEMPLATE.html OUTPUT.html",
  );
}

const markdown = fs.readFileSync(path.resolve(inputPath), "utf8");
const template = fs.readFileSync(path.resolve(templatePath), "utf8");
const mainStart = template.indexOf("<main>");
if (mainStart < 0) {
  throw new Error(`Template does not contain <main>: ${templatePath}`);
}

const head = template.slice(0, mainStart + "<main>".length);
const output = `${head}\n${renderMarkdown(markdown)}</main>\n</body>\n</html>\n`;
fs.writeFileSync(path.resolve(outputPath), output, "utf8");
