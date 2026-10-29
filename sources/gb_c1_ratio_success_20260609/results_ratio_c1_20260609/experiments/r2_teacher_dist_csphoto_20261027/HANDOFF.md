# CS/Photo five-seed DIST extension

## COMPLETE_AUDITED

2026-10-27完成，当前无队列或拟合待续。60新DIST学生/12000更新，两图短smoke和全部模型/指标/统计审计通过。queue_status=COMPLETE，锁已释放。报告入口REPORT_zh.md、research/r2_teacher_dist_csphoto_20261027/RESULT_INTERPRETATION_zh.md。

CS/Photo完整学生带噪accuracy92.470±.197/84.842±.379%，比CE学生+.676±.100/+2.084±.800pp；比完整教师+.728/+1.836pp。新五图完整学生73.692±.488%，比CE学生+2.071±.136pp，比自身教师+1.170±.186；clean比CE学生accuracy+.124pp。300合并指标/120新CPU重载/288meanSD及额外492paired/60transfer通过。

论文最新五图表在tables_five_graphs/{accuracy,macro_f1}.tex与CSV；旧三图表不改。无稿件/GitHub更改或新增实验。以下为历史运行说明，勿重复启动gpu_queue.py。

Approved scope: Coauthor-CS and Amazon-Photo x seeds0..4 x clean/U0.4/U0.6 x R2_CE/R2_FULL =60newstudents,12000updates. Existing three-graph five-seed DIST runtime read-only. New fullfivegraph tables150students/150teachers, no Pair0.4 requested forDIST.

Code here. Runtime D:/codex_temp/r2_teacher_dist_csphoto_20261027. Read queue_status.json before any launch. CPU-onlypreflight validates60teachers. GPUqueue standardlibwaits for otherPythonowners, including the active ssl_backbone_screen_20261026 worker60708 andvenv41452; never stop or modifythem. Parent/childIDs maychange; queue inventories currentprocesses. No calendarautomation.

Commands (workspace cwd): D:/codex_temp/noisygl-rocm721/Scripts/python.exe -B experiments/r2_teacher_dist_csphoto_20261027/run.py --stage preflight; afterpassing, gpu_queue.py once. queue.lock/run.lock forbidduplicates. WindowsStart-Process uses hiddenwindow andnewD stdout/stderrlogs. Queue invokes run.py all -> audit.py -> COMPLETE or FAILED. If WAITING_GPU, no studenttraining yet. If FAILED inspect background_stderr.log/failure_latest before recovery; do notdeleteexistinglockswithoutverifyingprocesses.

5contracttests passed; redFileNotFound beforeimplementation preserved. Source/inputhashesandteacherfreeze are authoritative inD. No formalGPUfit occurred duringCPUpreflight. All60newchoices freezebeforetest, report/tables afterallfits. Previousresultsunchanged. No newteachersearch/SSL/weights/options/manuscript/GitHubchanges.

Final outputs aftercompletion: REPORT_zh.md, audit.json, complete.json, tables_five_graphs/{summary,paired_summary,results}.csv andaccuracy.tex/macro_f1.tex. copythinreportstocodefolder; checkpoints/probabilities/logsinD. Currentexecutionstatealwaysqueue_status.json,preflight_complete.json,andstatus.json.
