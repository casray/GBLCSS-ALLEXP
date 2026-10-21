# Third-party notices

## NoisyGL

The dataset split, synthetic-noise protocol, and matched noisy-label baselines were adapted from NoisyGL:

> Copyright (c) 2024 NoisyGL

NoisyGL is distributed under the MIT License. Its copyright and permission notice are retained here as required by that license. The upstream project is available at <https://github.com/eaglelab-zju/NoisyGL>.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## GRACE

The GRACE encoder and contrastive objective follow the public GRACE implementation associated with *Deep Graph Contrastive Representation Learning*. The adapted implementation in `src/gblcss/grace.py` is distributed under the Apache License 2.0 terms of the upstream project: <https://github.com/CRIPAC-DIG/GRACE>. A copy of the license is provided in `licenses/GRACE-APACHE-2.0.txt`.

## PosteL

The no-IPL target construction in `src/gblcss/baselines/structure.py` follows `relabel.py` in <https://github.com/ml-postech/PosteL>. Its upstream MIT notice is retained below. The MLP training protocol is the adaptation described in the manuscript.

Copyright (c) 2025 MLLab @ Postech

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Other comparison methods

ALS, SALS, Graph-MLP, GSDN, GLNN-style, FF-G2M, KRD and VQGraph use the experiment-specific implementations in `src/gblcss/baselines/`. The Graph-MLP adaptation retains the shared MLP and sampled edge-contrastive loss from NoisyGL's `GraphFreeFamily_Predictor.py`. Method-specific settings are in `configs/main/protocol_adaptations.yaml`.
