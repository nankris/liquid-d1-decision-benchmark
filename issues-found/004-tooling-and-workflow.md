# 004 - Tooling and workflow notes

- **Colab connector (googlecolab/colab-mcp) can't attach to an already-running notebook tab.** It opens its own notebook, so monitoring a run started by hand wasn't possible. Runs are now started and monitored in one browser session.
- **Wrong Google account.** The browser's default Google account for Colab was not the one with compute units. Always open Colab with the account that holds the units (`?authuser=` matters) and check the avatar before connecting.
- **Idle runtimes cost compute units.** Last notebook cell calls `runtime.unassign()`, and every run ends with "Disconnect and delete runtime" + a Manage sessions check, even on failure.
- **Results lost on unassign.** The results JSON lives in `/content`; it must be downloaded before the runtime is deleted.
- **Agents writing code (Codex, Grok CLI) still need human review.** Codex got a dataset label meaning wrong (issue 001-3); a review caught 9 real bugs in the GPU notebook (issue 002).
