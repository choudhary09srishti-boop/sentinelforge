# SentinelForge architecture

## Layers (backend/app/)

| Layer | Folder | Responsibility |
|---|---|---|
| HTTP | api/ | Thin routes, request/response schemas, API-key auth |
| Cross-cutting | core/ | Typed settings, request IDs, error handling, (later) audit and logging |
| Integration | mcp_server/, mcp_client/ | Tools, resource and prompt exposed over MCP; client used by API and agent |
| Knowledge | rag/ | Parse, chunk, embed, index, retrieve, cite, dedupe, delete |
| Brain | agent/, providers/ | LangGraph workflow; LLM adapters (demo mode, Groq via httpx) |
| Execution | sandbox/ | Isolated runs, path validation, resource limits, no network by default |
| State | storage/ | SQLite: documents, chunks, tasks, patches, audit events |

## Request flow

UI -> api/ -> mcp_client -> mcp_server tools -> rag/ | sandbox/ | repository inspection.
The agent also goes through mcp_client, and uses providers/ for model calls.

## Security principles

- MCP is the real integration layer, not decoration.
- Retrieved text is data, never instructions.
- Patches are never applied without explicit approval tied to a specific patch version.
- Generated code runs only inside the sandbox boundary.
- Secrets live in .env, never in frontend code, logs, or the repository.
- External LLM use is always disclosed in the API and UI.

## Build order

Phase 2 API skeleton (done here) -> 3 MCP -> 4 RAG -> 5 agent + providers -> 6 sandbox ->
7 Streamlit prototype -> 8 React frontend -> 9 quality and docs -> 10 deployment.
