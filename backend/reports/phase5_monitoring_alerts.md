# Phase 5 monitoring and alert policy

## Replay eligibility and audit result

The replay controller accepts only the chronological test partition recorded in a trained artifact's feature manifest. It checks the loaded dataset filename and content fingerprint, train/validation/test boundaries, train-only preprocessing fit metadata, model lifecycle status, and artifact readability before creating a replay. A request interval must be contained in the manifest's test period. Replay identity is deterministic from model/version, dataset identity and fingerprint, and requested UTC boundaries.

At this implementation's audit, the checked-in registry contains only legacy records for `demo_aditya_l1_solexs_helios.csv`; none has the required trained lifecycle state and complete verified artifact contract. The local pipeline is also demo-mode. Therefore no real held-out solar observations can be replayed here. The backend correctly returns structured blockers; it does not substitute training or synthetic rows. The currently reported live-feed state is unavailable.

When an eligible artifact is provided, each forecast is issued at an ordered observation timestamp. Its predictor receives only raw measurement columns up to and including that timestamp; reference labels remain in a separate outcome map and can be attached only after the horizon end is observed during replay. The persisted record distinguishes observation, issue, valid-window, processing, and matured outcome times. Replay state and inference records are stored in the database and duplicate replay issues are constrained by replay/model/issue time.

## Alert policy

The shared provisional bands are LOW `<0.30`, WATCH `0.30–<0.50`, MODERATE `0.50–<0.70`, HIGH `0.70–<0.85`, and CRITICAL `>=0.85`. Scores are uncalibrated classifier scores and these bands are operational display labels, not validated event probabilities.

Default settings (`provisional-v1`) are trigger `0.50`, clear `0.40`, cooldown 900 seconds, expiry 21600 seconds, and material score change `0.10`. A new alert requires the configured trigger and no saved negative model decision. An active alert is updated for every valid score; scores from clear through below trigger preserve its active state, and a score below clear resolves it. Expiry resolves an alert when its last update is at least the configured expiry age. A later trigger starts a new alert. Invalid or missing scores create no state transition and never clear an alert.

Alert identity is scoped by model, horizon, and source classification. Each prediction/horizon transition is idempotent. A partial unique active-key index prevents duplicate active alerts; SQLite takes a write reservation before checking the key, while row locking is used where supported. Notification adapters are called only after the state transaction commits. No delivery integration is configured in this repository, so API-created alerts report `not_configured`; tests inject a sender to verify only explicit confirmation increments delivery count. Meaningful notification updates are a higher risk band or an absolute score change of at least the configured material-change amount, subject to cooldown.

## Threshold selection and limitation

The selection objective is defined before comparing candidates: among thresholds with sufficient validation support, maximize event-level utility `5 * hits - 5 * misses - false_alarms`, with one unit representing an operational false-alarm cost and a five-unit missed-event cost; report event counts, alert rate per day, probability of detection (POD), false alarm ratio (FAR = false alarms / issued alarms), and probability of false detection (POFD = false alarms / actual non-event windows) separately. Candidate clear thresholds must be strictly below the selected trigger and be assessed by replaying validation sequences with the same cooldown and re-arm policy. The final test split is excluded from selection.

No operational thresholds are selected by this implementation because the installed model records have no verified real-data validation predictions with independent reference labels and sufficient event support. Consequently the configured values remain explicitly provisional and the alert configuration endpoint reports `provisional_validation_blocked`. Final test labels are not used for threshold tuning.

## Capability classification

**Historical replay only, currently blocked pending eligible data and a trained verified checkpoint.** This checkout has no connected live satellite feed or notification delivery adapter. It has not executed a real held-out inference replay, and does not claim production-ready monitoring.
