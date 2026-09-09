'use strict';
const metrics = {
  accuracy: {title: 'Accuracy', direction: '↑ Higher is better', values: [89.633333, 96.6, 96.766667], max: 100,
    description: 'Correct decisions across all 3,000 examples. Auto-0.4b-2 gets 2,903 right.'},
  false_approve: {title: 'False approve', direction: '↓ Lower is better', values: [6.067095, 4.1399, 3.140614], max: 10,
    description: 'Deny examples mistakenly approved, out of 1,401. Auto-0.4b-2: 44; auto-1b-bf16: 58; original: 85.'},
  false_deny: {title: 'False deny', direction: '↓ Lower is better', values: [14.133834, 2.75172, 3.314572], max: 20,
    description: 'Approve examples mistakenly denied, out of 1,599. Auto-0.4b-2: 53; auto-1b-bf16: 44; original: 226.'},
};
const modelNames = ['auto-0.4b', 'auto-1b-bf16', 'auto-0.4b-2'];
function selectButton(button, selector) {
  document.querySelectorAll(selector).forEach(item => {
    const active = item === button;
    item.classList.toggle('active', active);
    item.setAttribute('aria-pressed', String(active));
  });
}
document.querySelectorAll('[data-metric]').forEach(button => button.addEventListener('click', () => {
  const metric = metrics[button.dataset.metric];
  selectButton(button, '[data-metric]');
  const heading = document.querySelector('#metric-title');
  heading.replaceChildren(document.createTextNode(metric.title + ' '));
  const direction = document.createElement('span'); direction.textContent = metric.direction; heading.append(direction);
  document.querySelectorAll('.chart-row').forEach((row, index) => {
    row.querySelector('.bar').style.setProperty('--value', `${metric.values[index] / metric.max * 100}%`);
    row.querySelector('strong').replaceChildren(document.createTextNode(metric.values[index].toFixed(2)));
    const unit = document.createElement('span'); unit.textContent = '%'; row.querySelector('strong').append(unit);
  });
  document.querySelectorAll('.chart-axis span').forEach((span, i) => {span.textContent = `${i * metric.max / 4}%`;});
  document.querySelector('#chart').setAttribute('aria-label', `${metric.title}: ${modelNames.map((name, i) => `${name} ${metric.values[i].toFixed(2)} percent`).join(', ')}. Axis 0 to ${metric.max} percent.`);
  document.querySelector('#metric-description').textContent = metric.description;
}));
let agent = 'pi', os = 'unix';
const agents = {pi: 'Pi', opencode: 'OpenCode', hermes: 'Hermes'};
function updateInstall() {
  document.querySelector('#install-command').textContent = os === 'windows'
    ? `& ([scriptblock]::Create((irm https://auto.ssh.codes/install.ps1))) -Agent ${agent}`
    : `curl -fsSL https://auto.ssh.codes/install.sh | sh -s -- --agent ${agent}`;
  document.querySelector('#install-comment').textContent = `# Install Auto for ${agents[agent]}`;
  document.querySelector('#copy-status').textContent = '';
}
document.querySelectorAll('[data-agent]').forEach(button => button.addEventListener('click', () => {
  agent = button.dataset.agent; selectButton(button, '[data-agent]'); updateInstall();
}));
document.querySelectorAll('[data-os]').forEach(button => button.addEventListener('click', () => {
  os = button.dataset.os; selectButton(button, '[data-os]'); updateInstall();
}));
document.querySelector('#copy-command').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(document.querySelector('#install-command').textContent);
    document.querySelector('#copy-status').textContent = 'Copied. Paste into your terminal to install.';
  } catch {
    const selection = window.getSelection(); const range = document.createRange();
    range.selectNodeContents(document.querySelector('#install-command')); selection.removeAllRanges(); selection.addRange(range);
    document.querySelector('#copy-status').textContent = 'Command selected. Copy it using your keyboard.';
  }
});
let probes;
document.querySelector('#probe-select').addEventListener('change', async event => {
  try {
    if (!probes) {const response = await fetch('/probes.json'); if (!response.ok) throw new Error('Unavailable'); probes = await response.json();}
    const probe = probes[Number(event.target.value)];
    document.querySelector('#probe-request').textContent = probe.user_request;
    document.querySelector('#probe-call').textContent = `${probe.call.tool}(${JSON.stringify(probe.call.args, null, 2)})`;
    const badge = document.querySelector('#probe-decision');
    badge.textContent = probe.expected === 'approve' ? 'Approve' : 'Deny';
    badge.className = `decision ${probe.expected}`;
    document.querySelector('#probe-reason').textContent = probe.expected_reason;
  } catch {
    document.querySelector('#probe-reason').textContent = 'Could not load this scenario. Follow the model-card link to read all examples.';
  }
});
