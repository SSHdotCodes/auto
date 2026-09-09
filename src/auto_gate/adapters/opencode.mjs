import { judgeWith, openCodeContext } from "./common.mjs";

export function opencodePlugin(config, judge = judgeWith(config)) {
  return async function AutoPlugin({ client }) {
    return {
      "tool.execute.before": async (input, output) => {
        let result;
        try {
          // Read this session from the host on every call: handles resumes, forks, and parallel sessions.
          const response = await client.session.messages({
            path: { id: input.sessionID },
            throwOnError: true,
          });
          if (!Array.isArray(response.data)) throw new Error("No transcript");
          const context = openCodeContext(response.data);
          result = await judge({
            ...context,
            call: { tool: input.tool, args: output.args },
          });
        } catch {
          throw new Error(
            "Auto: context or local inference unavailable. Run auto doctor; call was blocked.",
          );
        }
        if (result.decision !== "approve")
          throw new Error(result.reason ?? "Auto requires review");
        // Do not overwrite OpenCode permission rules or auto-answer unrelated permission requests.
      },
    };
  };
}
