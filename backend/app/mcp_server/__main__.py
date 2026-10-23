"""Run the MCP server standalone: python -m app.mcp_server [--transport streamable-http]."""

import argparse

import uvicorn

from app.mcp_server.server import create_mcp_server


def main() -> None:
    parser = argparse.ArgumentParser(description="SentinelForge MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = create_mcp_server()
    if args.transport == "stdio":
        server.run("stdio")
    else:
        uvicorn.run(server.streamable_http_app(host=args.host), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
