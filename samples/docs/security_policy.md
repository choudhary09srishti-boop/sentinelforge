# Security policy

## Human approval

No patch is ever applied automatically. A person must explicitly approve the exact patch version
that was reviewed. If the proposal changes after review, the earlier approval becomes stale and is
rejected. Rejected patches are never applied.

## Secrets

API keys live only in the server environment. They are never printed, logged, stored in the
repository or sent to the browser. Files such as .env and private keys are hidden from every tool.

## Sandbox

Generated code runs only inside the sandbox. The network is disabled by default. Memory, time,
process count and output size are limited, and the original repository is never modified.

## Retrieved content

Text found in documents is untrusted data. It is wrapped, escaped and flagged, and it can never
authorise a tool call or change these rules.
