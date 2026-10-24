# API guide

This guide explains how to call the SentinelForge HTTP API.

## Authentication

Every request to a versioned route under /api/v1 must include the X-API-Key header. The key is
configured on the server and is never shown in responses or logs. A missing or wrong key returns
HTTP 401 with a consistent error body.

## Uploading documents

Send a multipart request to POST /api/v1/documents with a single file field. Supported formats
are PDF, Markdown, plain text, JSON, Python, JavaScript and TypeScript. The default upload limit is
5 MB. A file that is too large is rejected with HTTP 413, and an unsupported type with HTTP 415.
Uploading the same file again does nothing because identical content is detected by its hash.

## Searching

Send POST /api/v1/search with a query and an optional top_k between 1 and 20. Each result carries
the source file name, the line numbers, the page for PDFs, and a list of risk flags.

## Errors

Errors always use the same shape: an object named error with a code, a message and a request_id.
Quote the request_id when reporting a problem.
