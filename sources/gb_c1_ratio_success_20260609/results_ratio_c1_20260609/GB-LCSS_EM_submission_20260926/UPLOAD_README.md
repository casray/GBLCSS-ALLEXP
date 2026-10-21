# GB-LCSS 投稿文件说明

本目录是 2026-09-26 整理的提交文件集，不是一个需要整目录上传的附件。请按投稿系统的文件类型分别选择文件。

最新正文为 `main_corrected.pdf`，已修正公式（10）、（12）末行逗号。原 `main.pdf` 因被其他程序占用未能覆盖，请勿上传旧 PDF。`main_source.zip` 已同步更新。

## 文件用途

| 文件 | 用途 |
| --- | --- |
| `main_corrected.pdf` | 最新正文审稿 PDF，23 页；系统允许 PDF 时选择 Manuscript |
| `main_source.zip` | 正文 LaTeX 源文件，20 个文件，ZIP 根目录直接包含 `main.tex`；选择系统提供的 LaTeX source files 类型 |
| `supplementary.pdf` | 附录 A-I，14 页；选择 Supplementary material 或系统对应类型 |
| `highlights.tex` | 四条 Highlights 的可编辑文件；选择 Highlights，上传前确认该入口接受 `.tex` |
| `highlights.pdf` | Highlights 的 1 页预览；不能代替系统要求的可编辑文件 |
| `supplementary_configurations.zip` | 实验配置快照和运行记录；如提交，作为独立补充资料，而不是 LaTeX 源文件 |
| `supplementary_source.zip` | 附录可编辑源文件，18 个文件；本地保留或按编辑部要求提供，不要将 LaTeX 源文件误选为公开 Supplementary material |

系统的实际文件类型名称可能不同。如禁止上传正文 PDF，应解压 `main_source.zip`，按系统要求分别上传 LaTeX、参考文献、图和表格文件，并指定 `main.tex` 为主入口。不要把正文、附录、Highlights 三个独立入口当成一篇稿件一起编译。

## 已完成的整理

- 正文、附录各自的源文件 ZIP 内没有子目录，也没有额外的外层文件夹。
- `\input` 和 `\includegraphics` 已改为同层文件名；五张图的内容和格式均未改变。
- 正文源文件保留实际引用的表格、算法、五张 PDF 图片、`references.bib`、定制样式 `gblcss-numeric.bst` 和重新生成的 `main.bbl`。
- 未将 `.aux`、`.log`、`.out`、`.blg`、`.spl`、绘图工程、重复预览、历史修改记录、打包脚本放入源文件 ZIP。
- 标准的 `elsarticle` 类和公共宏包由 TeX 发行版提供，没有复制师兄稿件的导言区，也没有迁移到 CAS 模板。已有页面尺寸、字号、边距、行距及 frontmatter 保持不变。
- 初次整理未删除或覆盖原文件夹；本次公式标点修订已同步到该目录的正文和附录源文件。请以本提交目录中的 PDF 和源文件 ZIP 为上传版本。

## 编译与核对

正文通常使用 XeLaTeX、BibTeX、XeLaTeX、XeLaTeX；附录使用 XeLaTeX 两次，Highlights 使用 XeLaTeX。解压每个源文件 ZIP 到各自目录后编译。

初次整理时用现有 Tectonic 编译环境实际编译三份文件，并确认文字和逐页渲染与整理前一致。随后按导师意见统一正文及附录的行间公式标点：句末使用句号，后接 where 的公式（2）、（17）保留逗号；分段公式（10）、（12）在 cases 内最后一行末尾使用逗号，括号外不加句号。数学内容和编号未变。重新编译后正文、附录和 Highlights 分别为 23、14、1 页；正文参考文献为 51 篇，无未定义引用。附录已包含 Table I.1 续表提示斜体、Table F.3 科学计数法的最新修改。

编译仍有原稿已有的公共宏包注释编码警告、图片 PDF 版本提示，以及正文 2.22221 pt 的输出阶段 overfull 提示。本轮未修改这些与目录结构无关的项目，渲染对照未发生变化。

## 上传前仍由作者完成

这一轮只整理现有文件，不代表全部投稿手续已完成。当前正文没有利益冲突、CRediT 作者贡献、数据可用性或生成式 AI 使用声明；请按实际情况及投稿系统要求补充。AI 声明按此前约定由作者自行填写，未代拟。

请同时确认通讯作者信息、全部作者同意投稿，以及系统是否要求投稿信、作者简介和照片或独立声明表。没有真实材料的项目未生成占位附件。补充这些内容后，请重新编译正文，并同步替换 PDF 和源文件 ZIP 中对应文件。

## 官方依据

- [Elsevier LaTeX instructions](https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions)：EM 的 LaTeX 源文件须处于同一文件层级；PDF 和源文件的上传方式取决于期刊系统设置。
- [Editorial Manager 文件准备说明](https://www.elsevier.support/publishing/answer/how-do-i-prepare-my-files-for-submission-in-editorial-manager)：以期刊 Guide for Authors 和系统要求的文件类别为准。
- [Information Sciences 作者指南入口](https://www.sciencedirect.com/journal/information-sciences/publish/guide-for-authors)：本轮在线读取返回 403，未将未核实的期刊专属要求当作已满足。

本说明留供作者使用，不上传为论文或补充材料。
