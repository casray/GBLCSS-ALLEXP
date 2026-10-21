# 原始逐运行结果分片

Git 直连 TLS 失败，大文件连接器传输也失败，因此通过连接器上传可重组的原始 gzip 分片。不是删减后的结果；按 MANIFEST 中顺序连接即得到原始 raw_results.jsonl.gz。各分片、原始压缩包 SHA-256 和原始 Git blob SHA 均在 MANIFEST.json。

在本目录运行以下 Python 命令，将压缩包还原至父目录；存在文件时只核验，不覆盖不匹配文件：

```python
import pathlib, json, hashlib, gzip
p = pathlib.Path('.')
m = json.loads((p/'MANIFEST.json').read_text(encoding='utf8'))
chunks = []
for part in m['parts']:
    b = (p/part['file']).read_bytes()
    assert len(b) == part['bytes']
    assert hashlib.sha256(b).hexdigest() == part['sha256']
    chunks.append(b)
b = b''.join(chunks)
assert hashlib.sha256(b).hexdigest() == m['original_sha256']
assert len(gzip.decompress(b).splitlines()) == m['records']
target = p.parent/m['original_file']
if target.exists():
    assert target.read_bytes() == b
else:
    target.write_bytes(b)
print('verified:', target, m['records'], 'records')
```

原始日志、预测 NPZ、准备缓存仍在本机运行目录，没有声称这些大型文件全已上传。
