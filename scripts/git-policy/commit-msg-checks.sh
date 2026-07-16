#!/usr/bin/env bash
# pre-commit `commit-msg`-stage hook (ADR-012). $1 = commit message file.
# Runs the message-dependent repo-policy checks against the REAL commit message.
#
# Absolute checks (no override): conventional subject, AI-disclosure trailer.
# Overridable (a `Policy-override:` line in the message skips them; local dev may
# also `SKIP=commit-msg-checks git commit`): Red Zone spec/review references.
set -uo pipefail
lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$lib_dir/_lib.sh"

message="$(cat "$1")"
subject="$(printf '%s' "$message" | sed -n '1p')"
repo_root="$(git rev-parse --show-toplevel)"
changed="$(git diff --cached --name-only)"

# A merge in progress (MERGE_HEAD present) has a git-generated subject and no
# authored body — skip the message-FORMAT checks; content is still checked at the
# pre-commit stage via the staged diff (code-reviewer 2026-07-16).
is_merge=0; [ -f "$(git rev-parse --git-dir)/MERGE_HEAD" ] && is_merge=1

rc=0
if [ "$is_merge" -eq 0 ]; then
    pc_conventional "$subject" || rc=1
    pc_ai_disclosure "$message" || rc=1
fi
if ! printf '%s' "$message" | grep -qiE '^Policy-override:'; then
    pc_redzone_ref "$message" "$changed" spec   "$repo_root" || rc=1
    pc_redzone_ref "$message" "$changed" review "$repo_root" || rc=1
fi
exit $rc
