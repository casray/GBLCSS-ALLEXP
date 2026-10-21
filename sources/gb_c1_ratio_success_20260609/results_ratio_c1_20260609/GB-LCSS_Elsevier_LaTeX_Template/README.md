# GB-LCSS LaTeX 工程

本工程按 `GB-LCSS_英文稿0907 (1).docx` 转换，使用 Elsevier 的 `elsarticle` 类。

当前排版参照师兄提供的模板及其 PDF：10pt Computer Modern 正文字体、1.15 倍行距、Letter 纸张、左右 17mm / 上下 28mm 页边距，引用及链接颜色为 `#1597D0`。原模板写了 `8pt`，但该选项未实际生效，PDF 正文实测为 10pt，因此本工程显式设置 `10pt`。

## 在 Overleaf 中使用

1. 上传本工程 ZIP，选择“Upload Project”。
2. 将主文件设为 `main.tex`，编译器设为 **XeLaTeX**。
3. 点击 Recompile。修改公式、交叉引用或参考文献后，必要时再编译一次。

## 修改位置

- `main.tex`：标题、摘要、正文、图表、两段算法和附录 A–H。
- `highlights.tex`：单独的 Highlights 文件，可设为主文件独立编译；五条内容与 Word 一致。
- `references.tex`：63 条参考文献。正文使用 `\cite{ref1}` 等引用，条目内容与 Word 对应，暂时沿用 Word 的编号顺序。
- `figures/`：5 幅高清 PNG。图片通过相对路径嵌入，不依赖本机目录。
- `main.pdf`：本地编译得到的阅读稿。

公式和算法已转成原生 LaTeX，表格内容也可以直接编辑。表 4 的两种噪声率放在同一表内；较长的附录表允许跨页并重复表头。

这次只进行格式转换与编译处理，不调整实验数据、研究结论或文献内容。作者、单位和投稿声明不在输入 Word 中，尚需按实际信息填写。保留的 `.bst` 文件来自原模板，可在今后转换为 BibTeX 时选用；当前编译不依赖它们。
