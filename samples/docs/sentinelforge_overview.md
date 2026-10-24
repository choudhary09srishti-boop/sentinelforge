# SentinelForge overview

SentinelForge is a privacy-conscious AI coding platform. It combines document search, a coding
agent and a safe execution sandbox behind one web application and API.

## Components

The platform has five parts. The MCP server exposes tools for inspecting a repository and
searching documents. The retrieval system parses files, splits them into chunks and stores
embeddings locally. The agent plans a change and proposes a unified diff. The sandbox runs the
tests in an isolated workspace. The web app lets a person review everything.

## Data flow

A user uploads documentation. The files are parsed and chunked, and each chunk receives an
embedding computed on the local machine. When a question is asked, the system retrieves the best
chunks and shows citations with the file name and line numbers.

## Privacy

Document processing and embeddings stay on the server. Only when an external model provider is
enabled are prompts and retrieved snippets sent to that provider, and the interface says so.
