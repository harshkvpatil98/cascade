## What this changes

<!-- What broke, or what is new. Lead with the problem. -->

## Measured

<!-- The numbers. If this moves an acceptance criterion, give the before and
     after. If a criterion is missed, say so with the measured value and a
     diagnosis -- do not relax it. -->

| Quantity | Before | After | Criterion |
|---|---|---|---|
|  |  |  |  |

## Gates

- [ ] `make ci` — ruff, black, mypy strict, offline tests
- [ ] `make test-all` — integration and leakage against live services
- [ ] No invariant weakened, and no static check exempted
- [ ] Any new database change is a **new** forward-only migration
- [ ] `CLAUDE.md` build log updated if this closes or moves a milestone
- [ ] `README.md` updated if a number it quotes has changed

## Notes

<!-- Anything deferred, and why. "Blocked on a credential" is a real reason;
     "will fix later" is not. -->
