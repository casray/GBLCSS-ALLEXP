# 带噪单位化五图扩展

状态：36/36新头、12/12复现、120评分完整；CS/Photo复用。结论见REPORT_zh.md，完整表见results/。

运行根：D:/codex_temp/ssl_noisy_norm_extension_20261028。

从项目根目录复核：

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -B experiments/ssl_noisy_norm_extension_20261028/review/check_results.py
```

原始启动入口launch.py已有完成保护；本批已结束。后续新SSL方案需单独目录，clean不设当前否决门。
