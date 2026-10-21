# Frozen GRACE embedding manifest

本文五个核心数据集的主表结果在每个数据集内复用同一份冻结 GRACE 表征。以下 SHA256 用于投稿前结果追溯；公开 manifest 不记录本机绝对路径。

## 主表 anchor H

| Dataset | Shape note | SHA256 |
|---|---|---|
| Cora | 128-dimensional H | `5e72323b2dd82cb583d1704bd0d8af9ecb135eef9771826d56af249956c64416` |
| CiteSeer | 256-dimensional H | `0c76a21aef43f9b6856d06b1e35d7deb4a18edc017302963357da97e24bc2ed1` |
| CoraFull | 256-dimensional H | `1940d2f55754ccc70ccc3216b5c8e6399474983be4a93521339d5adbdbf07191` |
| Coauthor-CS | 256-dimensional H | `de7520acae6d1049ca907b5ec3b8ab634a6f1852f1653ab1df8b4b0f28d89456` |
| Amazon-Photo | 256-dimensional H | `596e8cf3ad57f820895eb2edfd85d416eb5e077a54075a11a62a09d7a07ff30b` |

## MLP-37 三预训练种子重训资产

以下 9 份 H 只用于预训练随机性分层实验，不替换上表主表 anchor。每个 cell 同时归档 encoder/projector checkpoint、optimizer 状态、逐 epoch loss、配置和成本元数据。

| Dataset | H seed | Shape | H SHA256 | Checkpoint SHA256 |
|---|---:|---:|---|---|
| Cora | 0 | 2708×128 | `68409af79939967c12234d9076f7689f86d3f2d9e1799ea2eb5f8fdbdf81826f` | `87ba06cc59d15c781386cb1c51a5bf471087fda9ce94079fcc3927188371a52c` |
| Cora | 1 | 2708×128 | `3726f7a2778875c444d5d031d26cc172b48120811a60aa889fd0bc5ea2c246e9` | `bea7469d40371d5a8a96427cbbe2476e1bc27bd6cdf9a718d89d06d01041dd79` |
| Cora | 2 | 2708×128 | `4c404eb0785417475d098bb2993fbdc140d3dba201b7064e3fc14fe63d1c645d` | `e923f59eafba1c4176fdb6c2a984630b9b1d6839d35358216b219c51a16ebef6` |
| CiteSeer | 0 | 3327×256 | `e04e3409c70645277431878380c1975c2af8bf7647e480d4833f2435e52f1d74` | `0941131317e6c89209dbbce8f6f43a1e70483ae20b618a4f945b95d1969e1619` |
| CiteSeer | 1 | 3327×256 | `5d9adee5d2e41a8e38aa2525f487adb6bf93c8e08e5a8967f8b85c339c203565` | `387d71df2857d92fdd31e47339b2e535fb4d290def1171afc5eefbffc2846a3d` |
| CiteSeer | 2 | 3327×256 | `359f9a3211cc3e6e6d0d671685308e8f103b29d709499decd42e10318b27025e` | `75d80f5194e8cb7f01d1eccec55bbf17307164c2a91dfd9b0caf05ce98fe8769` |
| CoraFull | 0 | 19793×256 | `17272becaa7fcce9ea5266377e55a2de16a2607cb7770069a7d130644181a66a` | `3283b904dca19ef1b2ac5f7385b3a4e7480934ff55fe380a13af74df27a8206b` |
| CoraFull | 1 | 19793×256 | `f6930443ccfa09dce81736030ac47c0570773661f9898cc599892ac3b57c3588` | `e9c95fe86836f882496254d09b5ed15ed428be7e348c66ca63314b5646653b53` |
| CoraFull | 2 | 19793×256 | `cd41e7e8ebea75aa7d291fc455bab808c742c4b96e5d6487aa2aecfa6c29811f` | `e03bdfa19d5cb697a9c5b3cfc58ce60ba91f157ffe1a7d966906ce09afe1e164` |

主表五种子误差条仍条件于上表对应单份 H；MLP-37 的 3×3 分层实验仅用于补充估计 Cora、CiteSeer 与 CoraFull 的预训练随机性，不把 n=3 表述为完整管线方差。
