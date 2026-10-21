# GB-LCSS: Pattern Recognition 排版版

日期：2026-09-29。基于 `GB-LCSS_EM_submission_20260926` 中的主稿和附录源文件包迁移，原包未修改。

## 文件

- `main.pdf`：CAS 主稿，26 页。
- `main_source.zip`：主稿的扁平结构 LaTeX 源文件包，入口 `main.tex`。
- `supplementary.pdf`：附录 A-I，17 页。
- `supplementary_source.zip`：附录的扁平结构 LaTeX 源文件包，入口 `supplementary.tex`。
- `highlights.docx`、`highlights.pdf`、`highlights.tex`：沿用已定稿的 Highlights，内容未改。
- `supplementary_configurations.zip`：沿用原配置归档。
- `main_source/`、`supplementary_source/`：本地可编辑工作目录。上传源文件时使用对应 ZIP，不必上传编译日志。

## 本次调整

1. 使用用户下载的 CAS 2.4（2024/05/04）中的 `cas-sc.cls` 和 `cas-common.sty`，不修改官方类文件。
2. 主稿使用单栏 `review` 模式，正文双倍行距，A4 页面；摘要、图注、表格等保留 CAS 的默认或原有局部排版。页面上下边距 25 mm、左右边距 20 mm。
3. 通讯作者改为 **Bing Huang**，邮箱 **huangbing@nau.edu.cn**。作者顺序、其他邮箱和单位不变。
4. 使用 Elsevier 的 `elsarticle-num.bst`，保留按正文首次引用排序的数字编号。CAS 示例中的作者-年份排序样式未采用。
5. 删除模板自动生成的空 ORCID 行，邮箱使用文字标签，无需外部图标文件。

## 编译

分别解压两个源文件 ZIP，在各自目录编译，不要混在同一目录。

主稿：XeLaTeX -> BibTeX -> XeLaTeX -> XeLaTeX。
附录：XeLaTeX -> XeLaTeX。
Overleaf 可选择 XeLaTeX；本次已使用 Tectonic（XeTeX + BibTeX）成功编译。

## 核对结果与范围

- 主稿 Introduction 至文末的源文本除参考文献样式命令外逐字一致；附录从 `\appendix` 开始逐字一致。
- 图表、算法、图片和参考文献数据库与原包一致；主稿仍为 51 篇参考文献。
- 未发现未定义的引用；已查看整稿页面缩略图及首页，未发现页面外溢的文字。
- 主稿首页仍有一条 CAS 关键词零宽定位盒产生的 overfull 提示；已检查首页实际渲染，关键词和摘要未重叠或越界。未修改官方类文件以隐藏该提示。
- BibTeX 对 5 条未设传统页码的条目提示 empty pages，保留原数据库，不编造页码。
- 本轮仅迁移论文模板及通讯作者，**未迁入原 Information Sciences cover letter**，避免误用于新期刊。投稿时需另备面向 Pattern Recognition 的 cover letter，并与投稿系统中的作者信息一致。

## 排版依据

Pattern Recognition 的 Elsevier 官方期刊页面说明投稿稿件应为单栏、双倍行距、20-35 页并标页码。本版主稿为 26 页。作者指南页面未能直接读取，提交时仍应以投稿系统的最新要求为准。

https://shop.elsevier.com/journals/pattern-recognition/0031-3203

这是本地说明文件，不作为论文正文上传。
