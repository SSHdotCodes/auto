"use strict";
const metrics = {
  accuracy: {
    title: "Accuracy",
    direction: "↑ Higher is better",
    values: [96.0312, 95.5902, 95.5224, 95.3528, 93.1479, 88.8399, 70.2171],
    max: 100,
    description:
      "Correct decisions across all 2,948 tool calls. auto-0.4b-3 gets 2,746 right; auto-3b-3 gets 2,831.",
  },
  false_approve: {
    title: "False approve",
    direction: "↓ Lower is better",
    values: [3.1139, 0.6369, 4.8832, 5.5202, 6.0156, 9.0587, 22.6469],
    max: 24,
    description:
      "Harmful calls approved, out of 1,413. auto-3b-3: 44; DeepSeek V4.1 Flash: 9; auto-0.4b-3: 85; auto-0.4b-2: 320.",
  },
  false_deny: {
    title: "False deny",
    direction: "↓ Lower is better",
    values: [4.7557, 7.8827, 4.1042, 3.8436, 7.6221, 13.0945, 36.3518],
    max: 40,
    description:
      "Authorized calls denied, out of 1,535. auto-3b-3: 73; DeepSeek V4.1 Flash: 121; auto-0.4b-3: 117; auto-0.4b-2: 558.",
  },
};
const modelNames = [
  "auto-3b-3",
  "DeepSeek V4.1 Flash",
  "Codex auto-review",
  "GLM-5.3 Flash",
  "auto-0.4b-3",
  "auto-150m-3",
  "auto-0.4b-2",
];
function selectButton(button, selector) {
  document.querySelectorAll(selector).forEach((item) => {
    const active = item === button;
    item.classList.toggle("active", active);
    item.setAttribute("aria-pressed", String(active));
  });
}
document.querySelectorAll("[data-metric]").forEach((button) =>
  button.addEventListener("click", () => {
    const metric = metrics[button.dataset.metric];
    selectButton(button, "[data-metric]");
    const heading = document.querySelector("#metric-title");
    heading.replaceChildren(document.createTextNode(metric.title + " "));
    const direction = document.createElement("span");
    direction.textContent = metric.direction;
    heading.append(direction);
    document.querySelectorAll(".chart-row").forEach((row, index) => {
      row
        .querySelector(".bar")
        .style.setProperty(
          "--value",
          `${(metric.values[index] / metric.max) * 100}%`,
        );
      row
        .querySelector("strong")
        .replaceChildren(
          document.createTextNode(metric.values[index].toFixed(2)),
        );
      const unit = document.createElement("span");
      unit.textContent = "%";
      row.querySelector("strong").append(unit);
    });
    document.querySelectorAll(".chart-axis span").forEach((span, i) => {
      span.textContent = `${(i * metric.max) / 4}%`;
    });
    document
      .querySelector("#chart")
      .setAttribute(
        "aria-label",
        `${metric.title}: ${modelNames.map((name, i) => `${name} ${metric.values[i].toFixed(2)} percent`).join(", ")}. Axis 0 to ${metric.max} percent.`,
      );
    document.querySelector("#metric-description").textContent =
      metric.description;
  }),
);
let agent = "detect",
  os = "unix";
const agents = {
  detect: "every agent on this computer",
  "claude-code": "Claude Code",
  codex: "Codex CLI",
  gemini: "Gemini CLI",
  cursor: "Cursor",
  copilot: "GitHub Copilot CLI",
  pi: "Pi",
  opencode: "OpenCode",
  hermes: "Hermes",
};
function updateInstall() {
  const windows = os === "windows";
  const flag =
    agent === "detect"
      ? ""
      : windows
        ? ` -Agent ${agent}`
        : ` -s -- --agent ${agent}`;
  document.querySelector("#install-command").textContent = windows
    ? `& ([scriptblock]::Create((irm https://auto.ssh.codes/install.ps1)))${flag}`
    : `curl -fsSL https://auto.ssh.codes/install.sh | sh${flag}`;
  document.querySelector("#install-comment").textContent =
    `# Install Auto for ${agents[agent]}`;
  document.querySelector("#copy-status").textContent = "";
}
document.querySelectorAll("[data-agent]").forEach((button) =>
  button.addEventListener("click", () => {
    agent = button.dataset.agent;
    selectButton(button, "[data-agent]");
    updateInstall();
  }),
);
document.querySelectorAll("[data-os]").forEach((button) =>
  button.addEventListener("click", () => {
    os = button.dataset.os;
    selectButton(button, "[data-os]");
    updateInstall();
  }),
);
document.querySelector("#copy-command").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(
      document.querySelector("#install-command").textContent,
    );
    document.querySelector("#copy-status").textContent =
      "Copied. Paste into your terminal to install.";
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector("#install-command"));
    selection.removeAllRanges();
    selection.addRange(range);
    document.querySelector("#copy-status").textContent =
      "Command selected. Copy it using your keyboard.";
  }
});
let probes;
document
  .querySelector("#probe-select")
  .addEventListener("change", async (event) => {
    try {
      if (!probes) {
        const response = await fetch("/probes.json");
        if (!response.ok) throw new Error("Unavailable");
        probes = await response.json();
      }
      const probe = probes[Number(event.target.value)];
      document.querySelector("#probe-request").textContent = probe.user_request;
      document.querySelector("#probe-call").textContent =
        `${probe.call.tool}(${JSON.stringify(probe.call.args, null, 2)})`;
      const badge = document.querySelector("#probe-decision");
      badge.textContent = probe.expected === "approve" ? "Approve" : "Deny";
      badge.className = `decision ${probe.expected}`;
      document.querySelector("#probe-reason").textContent =
        probe.expected_reason;
    } catch {
      document.querySelector("#probe-reason").textContent =
        "Could not load this scenario. Follow the model-card link to read all examples.";
    }
  });
