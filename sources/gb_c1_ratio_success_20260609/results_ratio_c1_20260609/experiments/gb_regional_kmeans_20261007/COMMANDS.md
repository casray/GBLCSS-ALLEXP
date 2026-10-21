# Commands and execution notes

All commands run from the workspace root with the existing GPU environment.

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -m unittest discover -s experiments/gb_regional_kmeans_20261007 -p test_kmeans.py -v
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' experiments/gb_regional_kmeans_20261007/run_kmeans.py prepare
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' experiments/gb_regional_kmeans_20261007/run_kmeans.py replay
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' experiments/gb_regional_kmeans_20261007/run_kmeans.py grid
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' experiments/gb_regional_kmeans_20261007/analyze.py
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' experiments/gb_regional_kmeans_20261007/diagnostics.py
```

The first prepare invocation failed before clustering because the existing runner
did not add the NoisyGL repository to sys.path. The local runner now adds the
verified previous.NOISYGL path before importing its embedding loader. This was
repaired before input freezing and before any student training. No frozen source
or old result was edited. Clustering emitted a joblib physical-core discovery
warning; threadpool_limits explicitly restricts numerical clustering to four
threads. No clustering- or storage-efficiency claim is made from these runtimes.

The GB replay passed with zero probability difference, identical validation
trajectory, and identical best epoch. Subsequent training uses frozen k-means
partitions, with R0/R1 references from the deterministic stability experiment.
