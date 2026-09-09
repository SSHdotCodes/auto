import {judgeWith, piContext} from './common.mjs';

export function piPlugin(config, judge = judgeWith(config)) {
  return function autoExtension(pi) {
    pi.on('tool_call', async (event, ctx) => {
      try {
        const context = piContext(ctx.sessionManager.getBranch());
        const result = await judge({...context, call: {tool: event.toolName, args: event.input}});
        if (result.decision === 'approve') return;
        if (result.decision === 'review' && ctx.hasUI) {
          const permitted = await ctx.ui.confirm('Auto · review needed',
            `${result.reason}\n\n${event.toolName}\n${JSON.stringify(event.input)}\n\nAllow this one call?`);
          if (permitted) return;
        }
        return {block: true, reason: result.reason ?? 'Auto blocked this call'};
      } catch {
        return {block: true, reason: 'Auto could not evaluate this call. Run auto doctor.'};
      }
    });
  };
}
