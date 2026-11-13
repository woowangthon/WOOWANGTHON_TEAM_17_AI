# Atomic preferences v2 contract

`interpret_personal_draft_v2` is opt-in; `interpret_personal_draft` and the v1 Gemini parser retain their existing behavior. V2 returns the v1-compatible `suggestions` and `needsWalkingAnswer`, plus `atomicPreferences` and private `analysisMeta`.

## Canonical response

```json
{
  "schemaVersion": 2,
  "sourceDigest": "<server SHA-256>",
  "status": "current",
  "conditions": [{
    "id": "ap-<digest-prefix>-001",
    "dimension": "food",
    "target": "초밥",
    "polarity": "prefer",
    "priority": "unspecified",
    "priorityBasis": "unspecified",
    "scope": "trip",
    "audience": "personal",
    "frequency": {"count": 1, "operator": "exact", "period": "trip"},
    "value": null,
    "unit": null,
    "approximate": false,
    "evidence": "여행 중 초밥 한번",
    "origin": "model"
  }],
  "unresolved": []
}
```

`sourceDigest` covers normalized original text and fixed conditions serialized as sorted UTF-8 JSON. The server recomputes it and every `id`; client-supplied versions are ignored during round-trip validation. A changed source should be marked `stale` until reinterpreted. User-authored draft edits may set `origin=user`, but must retain source-grounded evidence. IDs are regenerated after edits.

`dimension` is `activity|food|walking|visit_count|meal_plan|atmosphere|accessibility|other`; `polarity` is `prefer|avoid|limit`; `priority` is `required|preferred|unspecified`; `priorityBasis` is `explicit|unspecified|user`; `scope` is `trip|day|segment|unspecified`; `audience` is `personal|group|unspecified`. Frequency is null or `{count: integer, operator: exact|min|max|target, period: trip|day}`. `value` is null, a finite number, or a non-empty string. `unit` is null or a string. `approximate` is boolean. Unresolved records are `{code,evidence,message}`.

Models do not emit IDs, digests, or origin. `evidence` must be a verbatim source span, but a substring alone does not prove polarity, amount, frequency, or priority. Ambiguous interpretations remain unresolved. In particular, an unmarked `avoid` stays unspecified priority; only an explicit user statement supports a model-origin `required` condition. User edits can explicitly author a required condition.

## Model call contract

Both providers receive `build_model_input(original_text, conditions)`, exactly `{originalText,conditions}`, and return exactly `{conditions,unresolved,scheduleItems}`. A condition uses the canonical fields except `id` and `origin`. `scheduleItems` uses the existing `SCHEDULE_ITEM_SCHEMA`; each provider result is passed through `validate_extracted_schedule`, which strips model evidence and creates server-owned booking IDs. This preserves the existing factual booking projection without another model call.

`PREFERENCE_AI_PROVIDER` selects `gemini` (compatibility default), `qwen`, or `ensemble`; the supplied ensemble deployment explicitly selects `ensemble`. `QWEN_PREFERENCE_URL` defaults to `http://127.0.0.1:18117/v1/interpret`. `PREFERENCE_AI_STRICT=true` fails the ensemble if one provider fails. Calls run independently in parallel, once per selected model under a shared bounded deadline. If both fail, a typed safe `ProviderError` is raised. Partial ensemble output is marked degraded. With two healthy providers, only matching semantic conditions become actionable; polarity, priority, scope, audience, quantity, frequency and approximate flags are compared. A conflicting or omitted condition remains source-linked in `unresolved` and is excluded from automatic application. Booking comparison ignores server-generated IDs. Matching outputs are not proof of correctness and do not represent calibrated confidence.

Server guards also reject unsupported numeric counts/values and model-required claims without explicit source cues. These conservative rules verify limited properties, not complete natural-language entailment. A named date such as “둘째 날 세 곳” is not generalized to a daily rate: the current frequency schema has no specific-date field, so such cases remain unresolved or segment scoped without a daily rate.

Legacy suggestion projection retains the fields and schedule item shape expected by current callers. V2 never asks a required walking question (`needsWalkingAnswer=false`); missing or unclear walking limits remain absent/unresolved rather than inferred.

## Group helper

`build_group_condition_summary` accepts only member IDs in `confirmed_member_ids`. It preserves hard/soft priority and counts a member at most once per grouped condition. It is a summary only; it does not select places, dates, routes, or optimize an itinerary.
