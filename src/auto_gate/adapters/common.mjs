import { spawn } from 'node:child_process';

export function contentText(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.filter(p => p?.type === 'text' && !p.synthetic).map(p => p.text ?? '').join('\n');
}

export function judgeWith(config) {
  return payload => new Promise(resolve => {
    const review = () => resolve({decision: 'review', reason: 'Auto unavailable; review required'});
    const child = spawn(config.python, ['-m', 'auto_gate', 'bridge'], {
      env: {...process.env, AUTO_HOME: config.autoHome}, windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'], shell: false,
    });
    let output = '';
    const timer = setTimeout(() => { child.kill(); review(); }, config.timeoutMs ?? 145000);
    child.on('error', () => { clearTimeout(timer); review(); });
    child.stdout.on('data', data => {
      output += data;
      if (output.length > 65536) { child.kill(); clearTimeout(timer); review(); }
    });
    // Keep the pipe flowing without forwarding local log contents to the agent.
    child.stderr.resume();
    child.on('close', code => {
      clearTimeout(timer);
      try {
        const result = JSON.parse(output);
        if (code || !['approve', 'deny', 'review'].includes(result.decision)) return review();
        resolve(result);
      } catch { review(); }
    });
    child.stdin.on('error', () => {});
    child.stdin.end(JSON.stringify(payload));
  });
}

export function piContext(entries) {
  const users = [], history = [], calls = new Map();
  for (const entry of entries) {
    const message = entry.message;
    if (!message) continue;
    if (message.role === 'user') {
      const text = contentText(message.content);
      if (text) users.push(text);
    }
    if (message.role === 'assistant' && Array.isArray(message.content)) {
      for (const part of message.content) {
        if (part.type === 'toolCall') calls.set(part.id, {tool: part.name, args: part.arguments});
      }
    }
    if (message.role === 'toolResult') {
      const call = calls.get(message.toolCallId) ?? {tool: message.toolName ?? 'tool', args: {}};
      history.push({...call, result: contentText(message.content)});
    }
  }
  return {user_request: users.join('\n\n'), history, context_complete: users.length > 0};
}

export function openCodeContext(messages) {
  const users = [], history = [];
  for (const message of messages) {
    if (message.info?.role === 'user') {
      const text = contentText(message.parts);
      if (text) users.push(text);
    }
    for (const part of message.parts ?? []) {
      if (part.type === 'tool' && ['completed', 'error'].includes(part.state?.status)) {
        history.push({tool: part.tool, args: part.state.input ?? {},
          result: part.state.output ?? part.state.error ?? ''});
      }
    }
  }
  return {user_request: users.join('\n\n'), history, context_complete: users.length > 0};
}
