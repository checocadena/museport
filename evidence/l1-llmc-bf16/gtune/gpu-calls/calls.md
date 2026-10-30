All gpu_test calls of the GPU tuning run, in order (from each request's result.json; the port each one built is in its port/ folder).

| # | request | vendor | profile | llm.c's test | speed GPU | port ms/step | CUDA ms/step | ratio | cuBLAS interop in port |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 20261029T191321 | nvidia | yes | PASS | NVIDIA RTX 4000 Ada Generation | 572.9 | 107.4 | 5.33x | no |
| 2 | 20261029T193139 | nvidia |  | FAIL | NVIDIA RTX 4000 Ada Generation | 333.7 | 107.9 | 3.09x | no |
| 3 | 20261029T200636 | nvidia |  | FAIL | NVIDIA RTX A4500 | 599.5 | 96.5 | 6.21x | no |
| 4 | 20261029T201456 | nvidia |  | PASS | NVIDIA RTX 2000 Ada Generation | 502.0 | 186.7 | 2.69x | no |
| 5 | 20261029T201856 | amd |  | PASS | AMD Instinct MI350X | 592.0 |  |  | no |
| 6 | 20261029T202320 | nvidia | yes | PASS | NVIDIA L4 | 419.8 | 143.2 | 2.93x | no |
| 7 | 20261029T203132 | nvidia |  | PASS | NVIDIA RTX A4000 | 258.5 | 116.5 | 2.22x | no |
| 8 | 20261029T203951 | nvidia |  | PASS | NVIDIA L4 | 235.1 | 161.8 | 1.45x | no |
| 9 | 20261029T204249 | nvidia | yes | PASS | NVIDIA L4 | 223.7 | 143.6 | 1.56x | no |
| 10 | 20261029T205002 | nvidia |  | PASS | - | - |  |  | no |
| 11 | 20261029T205617 | nvidia |  | PASS | NVIDIA L4 | 200.1 | 142.4 | 1.40x | no |
| 12 | 20261029T205916 | nvidia | yes | PASS | NVIDIA GeForce RTX 4090 | 71.6 | 44.1 | 1.62x | no |
| 13 | 20261029T210706 | nvidia |  | PASS | NVIDIA GeForce RTX 4090 | 71.8 | 44.2 | 1.62x | yes |
| 14 | 20261029T210937 | nvidia | yes | PASS | NVIDIA RTX A4000 | 178.1 | 113.7 | 1.57x | yes |
| 15 | 20261029T211656 | nvidia | yes | PASS | NVIDIA RTX 4000 Ada Generation | 144.7 | 107.4 | 1.35x | yes |
| 16 | 20261029T212450 | nvidia |  | PASS | - | - |  |  | yes |
| 17 | 20261029T213034 | nvidia |  | PASS | NVIDIA RTX A4000 | 154.1 | 116.7 | 1.32x | yes |
| 18 | 20261029T213423 | amd |  | compile FAILED | - | - |  |  | yes |
| 19 | 20261029T215617 | amd |  | PASS | AMD Instinct MI350X | 63.0 |  |  | yes |
| 20 | 20261029T222142 | nvidia |  | PASS | NVIDIA GeForce RTX 4090 | 49.8 | 43.5 | 1.14x | yes |
| 21 | 20261029T222420 | nvidia | yes | PASS | NVIDIA L4 | 160.1 | 140.6 | 1.14x | yes |
| 22 | 20261029T223337 | nvidia |  | PASS | NVIDIA RTX 2000 Ada Generation | 205.5 | 181.9 | 1.13x | yes |
| 23 | 20261029T223624 | amd |  | PASS | AMD Instinct MI350X | 63.5 |  |  | yes |
