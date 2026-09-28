import http from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 3000);
const apiKey = process.env.OPENAI_API_KEY;
const collections = new Map();

if (!apiKey) throw new Error("OPENAI_API_KEY is required. Add it to .env.local.");

function send(res, status, value) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(value));
}

async function openai(endpoint, options = {}) {
  const response = await fetch(`https://api.openai.com/v1${endpoint}`, {
    ...options,
    headers: { Authorization: `Bearer ${apiKey}`, ...(options.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error?.message || `OpenAI request failed (${response.status})`);
  return payload;
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > 30 * 1024 * 1024) reject(new Error("Each upload must be smaller than 30 MB."));
      else chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function parseMultipart(buffer, contentType) {
  const match = contentType.match(/boundary=(?:"([^"]+)"|([^;]+))/i);
  if (!match) throw new Error("Missing upload boundary.");
  const boundary = Buffer.from(`--${match[1] || match[2]}`);
  const parts = [];
  let cursor = buffer.indexOf(boundary) + boundary.length + 2;
  while (cursor > boundary.length + 1 && cursor < buffer.length) {
    const next = buffer.indexOf(boundary, cursor);
    if (next === -1) break;
    const segment = buffer.subarray(cursor, next - 2);
    const split = segment.indexOf("\r\n\r\n");
    if (split !== -1) {
      const headers = segment.subarray(0, split).toString("utf8");
      const body = segment.subarray(split + 4);
      const name = headers.match(/name="([^"]+)"/i)?.[1];
      const filename = headers.match(/filename="([^"]*)"/i)?.[1];
      parts.push({ name, filename, headers, body });
    }
    cursor = next + boundary.length + 2;
  }
  return parts;
}

async function waitUntilIndexed(vectorStoreId, fileId) {
  for (let attempt = 0; attempt < 30; attempt += 1) {
    const item = await openai(`/vector_stores/${vectorStoreId}/files/${fileId}`);
    if (item.status === "completed") return;
    if (["failed", "cancelled"].includes(item.status)) throw new Error(item.last_error?.message || "This PDF could not be indexed.");
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error("Indexing is taking longer than expected. Please try again shortly.");
}

async function uploadPaper(req) {
  const raw = await readBody(req);
  const filePart = parseMultipart(raw, req.headers["content-type"] || "").find((part) => part.name === "paper");
  if (!filePart?.filename || !filePart.body.length) throw new Error("Choose a PDF to upload.");
  if (!/\.pdf$/i.test(filePart.filename)) throw new Error("Only PDF papers are supported.");

  const form = new FormData();
  form.append("purpose", "assistants");
  form.append("file", new Blob([filePart.body], { type: "application/pdf" }), filePart.filename);
  const file = await openai("/files", { method: "POST", body: form });

  const vectorStore = await openai("/vector_stores", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: "Papertrail session", expires_after: { anchor: "last_active_at", days: 1 } }),
  });
  await openai(`/vector_stores/${vectorStore.id}/files`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id: file.id }),
  });
  await waitUntilIndexed(vectorStore.id, file.id);
  const id = crypto.randomUUID();
  collections.set(id, { vectorStoreId: vectorStore.id, fileId: file.id, name: filePart.filename });
  return { id, name: filePart.filename };
}

function citationsFrom(response) {
  const seen = new Set();
  const citations = [];
  for (const item of response.output || []) {
    for (const content of item.content || []) {
      for (const annotation of content.annotations || []) {
        const fileName = annotation.filename || annotation.file_citation?.filename;
        if (fileName && !seen.has(fileName)) {
          seen.add(fileName);
          citations.push({ fileName, quote: annotation.text || "Source from uploaded paper" });
        }
      }
    }
  }
  return citations;
}

async function answerQuestion(req) {
  const { collectionId, question } = JSON.parse((await readBody(req)).toString("utf8"));
  const collection = collections.get(collectionId);
  if (!collection) throw new Error("Your paper session expired. Upload the PDF again.");
  if (!question?.trim()) throw new Error("Ask a question about the paper.");
  const response = await openai("/responses", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "gpt-5-mini",
      tools: [{ type: "file_search", vector_store_ids: [collection.vectorStoreId] }],
      include: ["file_search_call.results"],
      input: `You are a careful research assistant. Answer only from the uploaded paper. If it does not support an answer, say so plainly. Be concise, distinguish findings from interpretation, and cite the paper by filename when possible.\n\nQuestion: ${question.trim()}`,
    }),
  });
  return { answer: response.output_text || "I couldn't generate an answer from this paper.", citations: citationsFrom(response) };
}

const server = http.createServer(async (req, res) => {
  try {
    if (req.method === "GET" && (req.url === "/" || req.url === "/index.html")) {
      res.writeHead(200, { "Content-Type": "text/html" });
      return res.end(await readFile(path.join(__dirname, "index.html")));
    }
    if (req.method === "POST" && req.url === "/api/upload") return send(res, 200, await uploadPaper(req));
    if (req.method === "POST" && req.url === "/api/ask") return send(res, 200, await answerQuestion(req));
    send(res, 404, { error: "Not found" });
  } catch (error) {
    send(res, 400, { error: error.message || "Something went wrong." });
  }
});

server.listen(PORT, () => console.log(`Papertrail is ready at http://localhost:${PORT}`));
