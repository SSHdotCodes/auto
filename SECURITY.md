# Security

Auto is a probabilistic permission classifier. It can make false approvals and false denials, and cannot prove code safe. Keep native agent rules, human review for high-impact actions, and operating-system isolation appropriate to your workload.

The local daemon binds only loopback, uses a random per-run token, rejects browser origins, and avoids prompt logging. Model revisions and Transformers are pinned. Installers use isolated environments; loaders use absolute paths and do not interpolate tool arguments into shell commands. Missing context, invalid responses, timeouts, and model failures never silently approve a call.

This protects against accidental browser access and ordinary integration failures. It is not a boundary against malicious processes running as the same OS user, an administrator, a compromised coding agent, or another plugin that changes arguments after classification. Inputs from untrusted tools can attack the model itself; strong benchmark results are not a guarantee against prompt injection. Auto only checks actions that the host routes through its hooks.

Do not expose the local service, copy `runtime.json`, or attach credentials, private prompts, logs containing unrelated application data, or model-cache tokens to issues. `auto doctor` deliberately omits the bearer token. Auto's daemon log does not record requests, but review its contents before sharing it.

For a security vulnerability, use GitHub's **Report a vulnerability** feature on this repository. Share a minimal synthetic reproducer and the affected version. Use public issues for ordinary bugs without sensitive data. Current fixes target the latest release.
