import assert from "node:assert/strict";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { opencodePlugin } from "../../src/auto_gate/adapters/opencode.mjs";

const scratch = await mkdtemp(path.join(tmpdir(), "auto contract ü "));
process.env.PI_CODING_AGENT_DIR = path.join(scratch, "pi");
try {
  // Exercise the actual Pi TypeScript/Jiti extension loader, including a file URI with spaces/Unicode.
  const piMain = fileURLToPath(
    import.meta.resolve("@earendil-works/pi-coding-agent"),
  );
  const { loadExtensions } = await import(
    pathToFileURL(path.join(path.dirname(piMain), "core/extensions/loader.js"))
  );
  const source = pathToFileURL(
    path.resolve(
      path.dirname(fileURLToPath(import.meta.url)),
      "../../src/auto_gate/adapters/pi.mjs",
    ),
  ).href;
  const extensionPath = path.join(scratch, "zz-auto.ts");
  await writeFile(
    extensionPath,
    `import {piPlugin} from ${JSON.stringify(source)}; export default piPlugin({python: 'missing-auto-python', autoHome: ${JSON.stringify(scratch)}, timeoutMs: 2000});`,
  );
  const result = await loadExtensions([extensionPath], scratch);
  assert.deepEqual(result.errors, []);
  assert.equal(result.extensions.length, 1);
  const handler = result.extensions[0].handlers.get("tool_call")[0];
  const decision = await handler(
    { type: "tool_call", toolName: "bash", input: { command: "rm -rf /" } },
    {
      hasUI: false,
      sessionManager: {
        getBranch: () => [
          { message: { role: "user", content: "Read the README" } },
        ],
      },
    },
  );
  assert.equal(
    decision.block,
    true,
    "A real loaded extension must block when its bridge is unavailable",
  );

  // Use the real SDK to construct the transcript request, with an intercepted transport (no agent/LLM service).
  const { createOpencodeClient } = await import("@opencode-ai/sdk");
  let requested, received;
  const client = createOpencodeClient({
    baseUrl: "http://127.0.0.1:9999",
    fetch: async (request) => {
      requested = request.url;
      return new Response(
        JSON.stringify([
          {
            info: { role: "user" },
            parts: [{ type: "text", text: "Read README" }],
          },
        ]),
        { headers: { "Content-Type": "application/json" } },
      );
    },
  });
  const hooks = await opencodePlugin({}, async (payload) => {
    received = payload;
    return { decision: "deny", reason: "blocked by contract test" };
  })({ client });
  await assert.rejects(
    hooks["tool.execute.before"](
      { tool: "bash", sessionID: "session-contract", callID: "c" },
      { args: { command: "rm -rf /" } },
    ),
    /blocked by contract test/,
  );
  assert.match(requested, /\/session\/session-contract\/message$/);
  assert.equal(received.user_request, "Read README");
  assert.equal(received.call.args.command, "rm -rf /");
  console.log(
    "Pi 0.85.1 real extension loader and OpenCode 1.18.30 real SDK contracts passed.",
  );
} finally {
  await rm(scratch, { recursive: true, force: true });
}
