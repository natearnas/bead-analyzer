---
name: Post-release checklist
about: Track post-release housekeeping tasks
title: "Post-release checklist: vX.Y.Z"
labels: ["release"]
assignees: []
---

## Post-release checklist

- [ ] Pull latest `main` locally and verify clean state.
- [ ] Verify GitHub release is published and notes look correct.
- [ ] Verify tag points to the intended release commit.
- [ ] Run a quick GUI smoke test (`python -m bead_analyzer.gui`).
- [ ] Run a quick CLI smoke test on a known small stack.
- [ ] Confirm docs links/changelog render correctly on GitHub.
- [ ] Announce release to users/team channels.
- [ ] Create follow-up issue for next patch/minor milestone.

## Notes

- Release version:
- Release URL:
- Any follow-up fixes identified:
