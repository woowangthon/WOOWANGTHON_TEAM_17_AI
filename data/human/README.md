# Human evaluation annotations

The tracked annotation template is blank on purpose. It does not contain human labels. Copy it to `inbox.local.json`, add genuinely new cases, and keep the completed file local.

Each row needs a unique `sourceId` and a `sourceFamily` that groups related wording or situations. Use a new family for every independent case. Do not rewrite any synthetic or model-generated example. The importer checks exact source, family, ID, digest and normalized-text overlap with the current synthetic fixture. It also flags close whitespace-normalized character matches at ratio 0.82 or higher. These checks can catch direct copying and close wording; they do not establish semantic independence.

Write `target` in the same raw v2 format the runtime expects. Keep evidence grounded in the original request, preserve ambiguity in `unresolved`, and never invent numbers, priorities, or scope. The importer runs the runtime validator before sealing records.

For `reviewMetadata`, use distinct pseudonymous IDs such as `anon-author01` and `anon-reviewer01`; do not include names, email addresses, or phone numbers. `reviewedAt` must be ISO-8601 with a timezone offset. Set all three attestations to `true` only after a human wrote the source independently, it was not adapted from the synthetic/model data, and a second person reviewed the label. The importer verifies these statements were provided; it does not verify people’s identity or that the attestations are true.

`sourceOwner` and `sourceLicense` must state the real owner and the actual basis for using the source. The permissions object is structured: local evaluation must be explicitly allowed; `externalAiProcessing` and `publicRedistribution` must accurately reflect granted permission and default to `false`. These permission values are attestations too. Do not set them based on assumptions.

From the repository root, copy the template and import only after review:

```powershell
Copy-Item data/human/annotation-template.json data/human/inbox.local.json
python scripts/import_human.py --input data/human/inbox.local.json --output data/human/human-evaluation-v2.jsonl
```

The importer creates a `kind: human-evaluation`, `testOnly: true` manifest bound to the exact current synthetic JSONL SHA-256. The output rows are all `human_adjudicated` and `split: test`; the regular training loader rejects this extended human schema. If the synthetic fixture changes later, re-import and review the human annotations before using them. Keep annotation JSON, imported JSONL, and manifest out of Git.

Evaluation figures from these cases are separate from synthetic proxy metrics. Do not call them expert or population-level accuracy without an appropriate study design.
