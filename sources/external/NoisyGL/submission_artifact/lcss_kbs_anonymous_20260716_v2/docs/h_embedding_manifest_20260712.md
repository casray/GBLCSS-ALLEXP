# Frozen GRACE embedding manifest

本文五个核心数据集的全部下游种子复用同一份冻结 GRACE 表征。以下 SHA256 用于投稿前结果追溯；路径均相对于本项目运行环境记录，论文正文只需披露哈希与“同一数据集跨种子复用同一 H”的协议。

| Dataset | Shape note | SHA256 |
|---|---|---|
| Cora | 128-dimensional H | `5E72323B2DD82CB583D1704BD0D8AF9ECB135EEF9771826D56AF249956C64416` |
| CiteSeer | 256-dimensional H | `0C76A21AEF43F9B6856D06B1E35D7DEB4A18EDC017302963357DA97E24BC2ED1` |
| CoraFull | 256-dimensional H | `1940D2F55754CCC70CCC3216B5C8E6399474983BE4A93521339D5ADBDBF07191` |
| Coauthor-CS | 256-dimensional H | `DE7520ACAE6D1049CA907B5EC3B8AB634A6F1852F1653AB1DF8B4B0F28D89456` |
| Amazon-Photo | 256-dimensional H | `596E8CF3AD57F820895EB2EDFD85D416EB5E077A54075A11A62A09D7A07FF30B` |

限制：当前误差条覆盖下游 split/noise/training seeds，不覆盖 GRACE 预训练随机性。后续如重新预训练 H，必须生成新哈希并视为新的实验条件。
