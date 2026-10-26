# Preflight filesystem repair

No scientific or smoke fits had started. inputs.json atomic replacement raised
WinError5 while tracking historical artifacts. Preserve original protocol,
source manifest, tests, failure and preflight logs. Add bounded retry only for
PermissionError on atomic replace; retain failure after exhaustion. No change
to targets, fitting or selection.
