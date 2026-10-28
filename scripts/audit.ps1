$ErrorActionPreference = "SilentlyContinue"
$r = New-Object System.Collections.Generic.List[string]
function GG([string]$p, [string]$paths) { @(git grep -n -I -i -E $p -- $paths 2>$null) }
function Ok([string]$name, [bool]$cond, [string]$note = "") { $t = if ($cond) { "PASS" } else { "FAIL" }; $r.Add("[$t] $name $note") }
function Show([string]$name, $lines) { $r.Add("[INFO] $name"); foreach ($l in @($lines)) { $r.Add("       " + ([string]$l).Trim()) } }

$r.Add("=== REPO FILES ===")
foreach ($f in "Dockerfile",".env.example","AI_USAGE.md","README.md","requirements.txt","render.yaml","check.ps1",".github/workflows") { Ok "exists: $f" (Test-Path $f) }
Ok ".env not tracked" (@(git ls-files .env).Count -eq 0)
Ok ".env never in git history" (@(git log --all --oneline -- .env).Count -eq 0)
Ok "no Groq key in git history" (@(git log --all --oneline -G "gsk_[A-Za-z0-9]").Count -eq 0)
Ok "no Groq key in tracked files" ((GG "gsk_[A-Za-z0-9]{20,}" ".").Count -eq 0)
Ok "streamlit pinned in requirements.txt" ((GG "^streamlit==" "requirements.txt").Count -gt 0)

$r.Add("=== MCP ===")
$tools = GG "\.tool\(" "backend/app/mcp_server"
Ok "MCP tools >= 5 (bonus at 6+)" ($tools.Count -ge 5) ("count=" + $tools.Count)
Ok "MCP resource >= 1" ((GG "\.resource\(" "backend/app/mcp_server").Count -ge 1)
Ok "MCP prompt >= 1" ((GG "\.prompt\(" "backend/app/mcp_server").Count -ge 1)
Ok "MCP sandbox-execution tool exists" ((GG "sandbox|run_tests" "backend/app/mcp_server").Count -gt 0)
Ok "MCP client integration test" ((GG "list_tools" "backend/tests").Count -gt 0)
Ok "tool timeouts" ((GG "timeout" "backend/app/mcp_server").Count -gt 0)
Ok "health endpoint" ((GG "/health" "backend/app/api").Count -gt 0)
Show "MCP calls made by agent graph" @(git grep -h -o -E "tools\.call\([^,)]*" -- backend/app/agent/graph.py)
Show "agent graph edges" @(git grep -h "add_edge" -- backend/app/agent/graph.py)
Show "sandbox refs in agent graph" (GG "sandbox" "backend/app/agent/graph.py")

$r.Add("=== RAG ===")
foreach ($e in ".pdf",".md",".txt",".json",".py",".js",".ts") { Ok "parser supports $e" ((GG "`"$e`"" "backend/app/rag/parsers.py").Count -gt 0) }
Ok "content hashing" ((GG "sha256|content_hash" "backend/app/rag").Count -gt 0)
Ok "duplicate detection" ((GG "duplicate" "backend/app/rag").Count -gt 0)
Ok "deletion tool" ((GG "delete_document" "backend/app/mcp_server").Count -gt 0)
Ok "prompt-injection flags" ((GG "risk_flags" "backend/app/rag").Count -gt 0)
Ok "prompt-injection tests" ((GG "injection" "backend/tests").Count -gt 0)
Show "eval-related files" @(git ls-files | Select-String -Pattern "eval")

$r.Add("=== AGENT / SANDBOX ===")
Ok "sandbox: network off" ((GG "unshare-all|unshare-net" "backend/app/sandbox/runner.py").Count -gt 0)
Ok "sandbox: time limit" ((GG "timeout" "backend/app/sandbox/runner.py").Count -gt 0)
Ok "sandbox: memory limit" ((GG "memory|ulimit|rlimit" "backend/app/sandbox/runner.py").Count -gt 0)
Ok "sandbox: process limit" ((GG "ulimit|nproc|pids" "backend/app/sandbox/runner.py").Count -gt 0)
Ok "sandbox: output limit" ((GG "MAXOUT|max_output" "backend/app/sandbox/runner.py").Count -gt 0)
Ok "sandbox: command/file allowlist" ((GG "allowlist|allowed|ALLOWED" "backend/app/sandbox").Count -gt 0)
Ok "audit events" ((GG "audit" "backend/app").Count -gt 0)
Ok "request id" ((GG "request_id" "backend/app/core").Count -gt 0)

$r.Add("=== WEB / API ===")
Ok "HTTP rate limiting" ((GG "slowapi|rate_limit|ratelimit|RateLimit" "backend/app").Count -gt 0)
Ok "CORS configured" ((GG "CORSMiddleware" "backend/app").Count -gt 0)
Ok "upload size limit" ((GG "max_upload_mb" "backend/app/api").Count -gt 0)
Ok "Ollama / llama.cpp local provider" ((GG "ollama|11434|llama.cpp" "backend/app/providers").Count -gt 0)
Ok "UI mentions provider / external" ((GG "external|provider" "ui").Count -gt 0)
Ok "sample-data bootstrap at startup" ((GG "bootstrap|seed" "backend/app").Count -gt 0)

$r.Add("=== TESTS ===")
Show "pytest collected" @(& ".\.venv\Scripts\python.exe" -m pytest --collect-only -q 2>$null | Select-Object -Last 2)
$sec = @(git grep -h "def test_" -- backend/tests/test_security.py backend/tests/test_safety.py backend/tests/test_sandbox.py backend/tests/test_apply.py).Count
$r.Add("[INFO] security-focused test functions (4 files): $sec")

$r.Add("")
$r.Add("FAIL count: " + @($r | Where-Object { $_.StartsWith("[FAIL]") }).Count)
$r
$r | Set-Clipboard
