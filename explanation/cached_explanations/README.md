# Cached fallback explanations

Pre-generate a `.txt` file per demo image here (named `<image_id>.txt`) so the live demo
has a working fallback if the explanation API is down or rate-limited (Phase 2, step 3;
re-tested in Phase 7, step 1).

Example: `fabric_demo_01.txt` containing a plain-language explanation for that specific
demo image's flagged defect.
