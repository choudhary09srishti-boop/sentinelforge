# Retrieval evaluation results

- Run at: 2026-10-24T20:05:39+00:00
- Embedding model: `BAAI/bge-small-en-v1.5` (local, CPU)
- Mode: `hybrid`, top_k=3
- Indexing time for the sample corpus: 7.1s
- hit@1 = 0.88, hit@3 = 1.00, MRR = 0.92

| # | Question | Expected source | Rank found | Top sources |
|---|---|---|---|---|
| 1 | How do I authenticate requests to the API? | api_guide.md | 1 | api_guide.md, security_policy.md, api_guide.md |
| 2 | What happens if I upload a file that is too large? | api_guide.md | 1 | api_guide.md, settings.json, sentinelforge_overview.md |
| 3 | Who has to approve a patch before it is applied? | security_policy.md | 1 | security_policy.md, security_policy.md, sentinelforge_overview.md |
| 4 | What should I do first when the service goes down? | incident_runbook.pdf | 1 | incident_runbook.pdf, incident_runbook.pdf, api_guide.md |
| 5 | How do I respond to a leaked credential? | incident_runbook.pdf | 1 | incident_runbook.pdf, incident_runbook.pdf, api_guide.md |
| 6 | What is the maximum time a tool call may run? | settings.json | 3 | security_policy.md, rate_limiter.js, settings.json |
| 7 | Which release added support for TypeScript files? | release_notes.txt | 1 | release_notes.txt, api_guide.md, security_policy.md |
| 8 | Function that retries a failed request with growing delays | retry.ts | 1 | retry.ts, api_guide.md, api_guide.md |
