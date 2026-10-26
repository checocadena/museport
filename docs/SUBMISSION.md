# Submission notes (draft)

## Opening line of the 3-minute video (required: what was pre-existing, what was built)

> "Pre-existing: my open-source oneapi-agent-toolkit, the SYCL tools, agent loop and GPU test
> pipeline, used here as a dependency. Built this week: Muse Port. Meta's Muse plans a CUDA
> repository's migration, ports it kernel by kernel against checks it never sees, and the
> ports run on NVIDIA GPUs."

The demo should then show only this repository's work:
- `museport scan` and `museport plan` on llm.c;
- `museport port` with a hidden check failing, then passing;
- the variant check catching a copied implementation;
- GPU results, then the results table.

## Repository
- Public, created at the event: https://github.com/checocadena/museport.
- Contains only hackathon work. The toolkit is a pinned pip dependency and none of its code
  or history is in this repository. Third-party llm.c code in `evidence/` carries its MIT
  notice ([`THIRD_PARTY.md`](../THIRD_PARTY.md)).

## Numbers to quote (all from committed evidence; see the README table)
- 19 llm.c kernel files ported by `muse-spark-1.3`.
- 17 correct with zero migration helpers, by checks the model never sees.
- 14 pass on NVIDIA GPUs against llm.c's original CUDA.
- Muse API cost $5.51 for all 19.
- Limits to state: fp32 only, reduced sizes, one run per item, AMD not run.
