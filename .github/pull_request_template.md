## Summary

- what tool, registry, or site change is being proposed
- why it belongs in the central registry

## Registry Checklist

- [ ] Added or updated `registry/tools/<tool-id>/tool.yaml`
- [ ] Added or updated `registry/tools/<tool-id>/examples.jsonl`
- [ ] Confirmed license and source repository are public
- [ ] `parent_id` is valid and present in `registry/categories.yaml`
- [ ] Ran `python3 scripts/build_registry.py`

## Verification

- [ ] Confirmed generated files changed as expected
- [ ] Confirmed the site still loads registry data

## Notes

- risks, follow-ups, or moderation concerns
