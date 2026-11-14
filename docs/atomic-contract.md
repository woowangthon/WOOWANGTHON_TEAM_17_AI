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

Both extractors receive `build_model_input(original_text, conditions)`, exactly `{originalText,conditions}`, and return `{conditions,unresolved,scheduleItems}`. A condition uses the canonical fields except `id` and `origin`. Each booking candidate is grounded using `validate_extracted_schedule`, keeping its source quote private until date, time and venue have been reconciled. The public booking projection has server-owned IDs and no evidence field.

`PREFERENCE_AI_PROVIDER` selects `gemini` (default), `qwen`, or `ensemble`. Set `QWEN_PREFERENCE_URL` to the private serving endpoint; its default is `http://127.0.0.1:18117/v1/interpret`. The ensemble calls both extractors in parallel under one deadline. A malformed sibling atom is isolated. With no usable extraction the request fails. With one usable provider it returns grounded fields with `degraded=true`, or fails if `PREFERENCE_AI_STRICT=true`.

In ensemble mode, source occurrences and their subjects anchor each condition. Different days can both mention the same place. The server combines compatible fields and checks the local wording for polarity, units, counts and scope. One-sided supported facts can be applied. Remaining competing field values are offered anonymously to Gemini in at most one private structured review within the original deadline. It can select an offered value with a verbatim supporting phrase, or abstain. Missing, duplicated, invalid, unsupported and late decisions leave those fields open. Tests can inject independent providers and a reviewer without network access. The trained Qwen adapter and the extraction prompt are unchanged.

Before field reconciliation, competing targets with identical evidence in the same passage and dimension can share one claim when exactly one target is source-anchored and the other is a strict normalized prefix extension. The existing target check keeps the source-supported alternative. Distinct source-anchored subjects, unrelated paraphrases, bookings and different occurrences stay separate.

Negated necessity (for example, “먹어도 좋지만 필수는 아니에요”) does not erase an independently stated positive preference or exclusion. These narrow optionality phrases are ignored only while checking direction and hard-importance cues; original evidence stays intact. Negated necessity alone does not establish a preference. Genuine uncertainty, actual menu negation/exclusion and separate explicit allergy/required reasons retain their existing guards.

The BE exposes `suggestions.atomicPreferences.conditions` and `suggestions.atomicPreferences.unresolved` to the owner's v2 client. For example, an unresolved frequency can leave an accepted food atom with `frequency=null` and a note `{"code":"ambiguous_quantity","evidence":"초밥 한 번 먹고 싶어","message":"‘초밥’의 횟수·수량을 원문에서 확정하기 어려워요."}`. Notes use the existing `{code,evidence,message}` contract. Several conditions can share a quote. `ambiguous_*` notes and older provider notes can both appear. A missing target, unresolved polarity or an unsafe partial limit can leave only a note. `analysisMeta.reviewRequired` indicates notes exist. The HTTP save/receipt and stale-document contract stays at schema version 2. The BE reconstructs notes from trusted analysis, so omitting a note from a client PUT does not persist its dismissal. FE must distinguish merely hiding a note from a persisted edit.

Server guards also reject unsupported numeric counts/values and model-required claims without explicit source cues. These conservative rules verify limited properties, not complete natural-language entailment. A named date such as “둘째 날 세 곳” is not generalized to a daily rate: the current frequency schema has no specific-date field, so such cases remain unresolved or segment scoped without a daily rate.

Legacy suggestion projection retains the fields and schedule item shape expected by current callers. V2 never asks a required walking question (`needsWalkingAnswer=false`); missing or unclear walking limits remain absent/unresolved rather than inferred.

## Group helper

The downstream BE route adapter uses confirmed atomic fields directly. It retains `required` versus soft preferences and day versus trip frequency through candidate scoring and final route validation. Current atomic members bypass their lossy legacy suggestion projection during this internal generation path.

The existing candidate scheduler accepts optional `scheduleContext.preferenceRules`, a list of `{contentids,period,operator,count,required}` bounded by the existing request-byte limit. The BE deduplicates identical prompt rules across members. IDs must belong to the provided candidate pool, periods are `day|trip`, operators are `exact|min|max|target`, and counts are integers from 0 to 365. `target` is soft. This context contains no member references, private evidence, model outputs or original request. The normal drafter forwards these rules to Gemini and retains its existing unverified-proposal response. BE independently rechecks the scheduled plan and every optimization evaluation. Empty or omitted rules retain the existing scheduling contract.

`build_group_condition_summary` accepts only member IDs in `confirmed_member_ids`. It preserves hard/soft priority and counts a member at most once per grouped condition. It is a summary only; it does not select places, dates, routes, or optimize an itinerary.
