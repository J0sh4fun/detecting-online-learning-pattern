# Data Collection Guide

This model is a posture classifier, not a person identifier. The dataset must contain enough variation that the same label means the same behavior across different people, cameras, lighting, and seating positions.

## Labels

Collect only these labels for the posture model:

- `Focused`
- `Slouching`
- `Leaning on Desk`
- `Looking Away`

Do not collect `Using Phone` in this CSV. Phone use should stay as the YOLO path. Do not collect `Absence` unless you intentionally add a separate absence model.

## Frame Setup

Use the same setup rules for every person:

- Camera should show head, both shoulders, and at least the upper chest.
- Keep the face and shoulders inside the frame for the whole capture.
- Use normal room lighting. Avoid strong backlight from windows.
- Keep the camera stable. Do not move the laptop or webcam during a capture group.
- Sit at a natural online-class distance, usually arm length from the camera.
- Collect with and without visible wrists/hands. The previous dataset had many missing wrist samples, which made the model depend too much on missing-hand patterns.
- Avoid exaggerated acting. The goal is realistic student behavior.

## Per-Pose Requirements

### Focused

Use this when the student is attending normally.

Required:

- Face generally points toward the screen/camera.
- Shoulders are visible and roughly level.
- Neck is upright or only mildly tilted.
- Hands may be visible or not visible.
- Allow natural small motion: blinking, slight head movement, small posture shifts.

Do not label as `Focused` if:

- The person is clearly looking away from the screen.
- The head is resting on a hand, arm, table, or desk.
- The upper back is clearly rounded forward.

Important boundary examples:

- Collect focused frames where the person looks slightly down at the screen.
- Collect focused frames where the person looks slightly left/right but is still attending.
- These examples help prevent `Focused` from being predicted as `Looking Away`.

### Slouching

Use this when the student remains seated but posture collapses.

Required:

- Shoulders and upper back move forward or downward.
- Neck/head position drops relative to shoulders.
- Person is still looking generally at the screen or work area.
- No head support from hand/table.

Do not label as `Slouching` if:

- The head is resting on the desk or arm. Use `Leaning on Desk`.
- The main behavior is turning away. Use `Looking Away`.

Collect variation:

- Mild slouch, medium slouch, and clear slouch.
- Hands visible and hands not visible.
- Different camera distances.

### Leaning on Desk

Use this when the student is physically supported by the desk, arm, or hand.

Required:

- Head or upper body leans strongly toward the desk.
- Head may rest on hand, forearm, or table.
- One or both hands may be near the face.
- The pose should look inactive or disengaged, not just mild forward posture.

Do not label as `Leaning on Desk` if:

- The person is simply sitting with rounded shoulders. Use `Slouching`.
- The person is upright but looking to the side. Use `Looking Away`.

Collect variation:

- Head on left hand.
- Head on right hand.
- Chin on hand.
- Forehead/head down toward table.
- One wrist visible, both wrists visible, and no wrists visible.

### Looking Away

Use this when attention is directed away from the screen.

Required:

- Face/head clearly turns away from screen/camera, or eyes/head are directed off-screen for more than a moment.
- Body may stay upright.
- Shoulders remain visible when possible.

Do not label as `Looking Away` if:

- The person only glances slightly down at the screen. Use `Focused`.
- The person is resting on the desk. Use `Leaning on Desk`.
- The person is mainly slouched but still looking at screen. Use `Slouching`.

Collect variation:

- Looking left.
- Looking right.
- Looking upward.
- Looking down away from screen.
- Mild, medium, and strong turns.

## Collection Targets

Minimum useful target:

- At least 4 people.
- At least 300 cleaned rows per label per person.
- At least 1,200 cleaned rows per person.

Better target:

- 6 to 10 people.
- 500 to 800 cleaned rows per label per person.
- Similar row counts for every label and person.

For each person, collect multiple short groups instead of one long recording:

- 6 to 10 capture groups per label.
- 8 to 15 seconds per group.
- Change one condition between groups: distance, lighting, chair position, hand visibility, or pose intensity.

The updated collector saves at 4 FPS. A 10 second group gives about 40 rows. Ten groups per label gives about 400 rows before cleanup.

## Efficient Collection Workflow

1. Prepare the camera and lighting.
2. Confirm head, shoulders, and upper chest are visible.
3. Start `data_collection.py`.
4. Press the label key.
5. Wait through the countdown.
6. Hold the pose naturally for 8 to 15 seconds.
7. Press the same key again to stop.
8. Relax, change one condition, and repeat.

Keys:

- `0`: `Focused`
- `1`: `Slouching`
- `2`: `Leaning on Desk`
- `3`: `Looking Away`
- `q`: quit

Avoid collecting while transitioning into or out of the pose. Start the label only when the person is ready.

## Accuracy Rules

- Do not mix labels inside one capture group.
- Do not collect a label if you are unsure. Skip ambiguous frames.
- Do not shuffle raw data before cleanup if you want group fallback behavior to remain useful.
- Keep `test.csv` from a different person as an external holdout. Do not train on it unless you create a new external holdout.
- Use the same label definitions for every person. Inconsistent labeling is worse than having fewer rows.

## After Collection

Run cleanup:

```powershell
python model_pipeline\sample_fix.py --dry-run
python model_pipeline\sample_fix.py
```

Then test generalization:

```powershell
python model_pipeline\train_model_test.py
```

Only save the model if the external-person metrics are acceptable:

```powershell
python model_pipeline\train_model_test.py --save-model
```

## What To Watch In Metrics

Prioritize:

- `f1_macro`
- `balanced_accuracy`
- Per-class recall
- Train/test F1 gap

Do not rely on training-data CV alone. The external-person test is the real signal.

Warning signs:

- Train CV near `0.99`, but external-person F1 much lower.
- `Focused` recall below `0.75`.
- Many `Focused` rows predicted as `Looking Away`.
- Large mismatch in no-visible-wrist percentage between training and test data.

## Feature Notes

New collection writes extra features beyond the original 8:

- shoulder width relative to frame width
- signed head offset from shoulder center
- signed nose offset from shoulder center
- absolute pitch/yaw
- face detected flag
- nose, shoulder, ear, and wrist visibility
- visible wrist count
- hand visible flag

Old CSV rows can still train, but those rows use default values for the new features. For the improved features to help, collect new data with the updated `data_collection.py`.
