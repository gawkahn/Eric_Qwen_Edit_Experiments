#!/usr/bin/env bash
# PreToolUse hook — reject `git commit -m "..."` invocations whose message
# lacks the `AI-disclosure:` trailer.
#
# Enforces global CLAUDE.md §7 and Critical Rules #6 mechanically so the
# discipline survives Sonnet context drift. See project CLAUDE.md
# §"Commit-time hooks".
#
# Exit 0 = allow; exit 2 = block with stderr message visible to Claude.
# Commits opened in an editor (no `-m` flag) are NOT blocked — the user
# authors the message interactively and can add the trailer themselves.

set -euo pipefail

input=$(cat)
tool_name=$(printf '%s' "$input" | jq -r '.tool_name // ""')

if [ "$tool_name" != "Bash" ]; then
    exit 0
fi

command=$(printf '%s' "$input" | jq -r '.tool_input.command // ""')

# Match `git commit` with or without a `-C <path>` flag.
if ! printf '%s' "$command" | grep -qE 'git[[:space:]]+(-C[[:space:]]+\S+[[:space:]]+)?commit'; then
    exit 0
fi

# Only check `-m`-style inline message commits; editor-style commits let
# the user author the message interactively.
if ! printf '%s' "$command" | grep -qE -- '(^|[[:space:]])-m([[:space:]]|$)'; then
    exit 0
fi

# Already has the trailer — pass.
if printf '%s' "$command" | grep -q 'AI-disclosure:'; then
    exit 0
fi

cat <<'EOF' >&2
BLOCKED: git commit is missing the AI-disclosure trailer.

Required per global CLAUDE.md §7 and Critical Rules #6. Add to the
commit message body (not subject):

    AI-disclosure: Claude (<tier>) authored; Grant reviewed.

Example:

    feat: short description

    Body paragraph explaining why.

    AI-disclosure: Claude (Sonnet 4.6) authored; Grant reviewed.
    Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

If this commit is genuinely human-only:

    AI-disclosure: none
EOF
exit 2
