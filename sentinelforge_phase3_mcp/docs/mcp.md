# MCP server

Built on the official Python SDK (mcp 2.x, `MCPServer`). All tools are read-only: nothing here
writes to the workspace. Run standalone with `python -m app.mcp_server` (stdio) or
`python -m app.mcp_server --transport streamable-http` (also serves GET /health).

| Tool | Purpose |
|---|---|
| list_repo_files | List a workspace directory |
| read_repo_file | Read up to 500 lines with line numbers |
| search_repo | Literal text search (no regex, rules out ReDoS) |
| repo_summary | File counts by extension, largest files |
| propose_patch | Unified diff + hashes. Never writes. Approval is separate |
| server_health | Status, workspace availability, provider mode |

Resource: `sentinelforge://policy/security`. Prompt: `plan_coding_task`.

Safety: paths are validated (no absolute paths, `..`, null bytes or symlink escapes), secrets
and VCS folders are hidden, file size and time are capped, errors are JSON with a stable `code`.
The HTTP API reaches every tool only through `app.mcp_client.McpGateway`.
