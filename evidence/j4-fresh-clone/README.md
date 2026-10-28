# J4: `llama2-sycl` from a fresh clone

This run followed the Quick start in [`llama2-sycl/README.md`](../../llama2-sycl/README.md)
exactly, on Oct 28. It started from a fresh `git clone` of this repository's `j4-llama2` branch
at `60b4e1f`, on the CPU device in the cloud session, at $0.

| Step | Result |
|---|---|
| `wget .../stories15M.bin` | the model, sha256 `cd590644...` as in the check |
| `make llama2` | `icpx -fsycl -O2 llama2.cpp -o llama2`, 7 s |
| `./llama2 stories15M.bin 0.0 64 "Once upon a time"` | the story, 225 bytes, at 174 tok/s on the CPU |
| `./llama2 stories15M.bin 0.0 64 "Tom and his dog went to the park"` | the story, 233 bytes, at 196 tok/s |

Both stories are identical, byte for byte, to llama2.cu's own output recorded on an NVIDIA GPU
([`compare.txt`](compare.txt)). Full session: [`session.log`](session.log).
