# Differential check: the original CUDA's recorded output as the hidden reference

Port run: `evidence/port-llmc-tier1-run2`

| item | recorded on | two runs identical | recording vs CPU reference | port vs CPU reference | port vs recorded CUDA |
|---|---|---|---|---|---|
| t1-crossentropy-forward | NVIDIA RTX A4500 | yes | PASS: 1/1 case(s) match (worst max abs err 1.19e-07, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 1.19e-07, atol 1e-05) |
| t1-crossentropy-softmax-backward | NVIDIA RTX 2000 Ada Generation | yes | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) |
| t1-permute | NVIDIA RTX A4500 | yes | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) |
| t1-residual-forward | NVIDIA RTX A4500 | yes | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) |
| t1-gelu-forward | NVIDIA RTX A4500 | yes | PASS: 1/1 case(s) match (worst max abs err 1.19e-07, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 1.19e-07, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 1.19e-07, atol 1e-05) |
| t1-gelu-backward | NVIDIA RTX 2000 Ada Generation | yes | PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05) |
| t1-encoder-forward | NVIDIA RTX A4500 | yes | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) |
| t1-encoder-backward | NVIDIA RTX 2000 Ada Generation | yes | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-05) |
| t1-adamw | NVIDIA RTX A4500 | yes | PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 0.0001) | PASS: 1/1 case(s) match (worst max abs err 1.79e-07, atol 0.0001) | PASS: 1/1 case(s) match (worst max abs err 1.79e-07, atol 0.0001) |
| t1-fused-residual-forward | NVIDIA RTX 2000 Ada Generation | yes | PASS: 1/1 case(s) match (worst max abs err 1.43e-06, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 5.96e-07, atol 1e-05) | PASS: 1/1 case(s) match (worst max abs err 1.43e-06, atol 1e-05) |
