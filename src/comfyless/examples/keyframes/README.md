# Example keyframes for `video-plan.json`

Six 1280×704 keyframes (Wan 720p class, /16-aligned) backing the four
segments of `../video-plan.json`. Authored 2026-07-20 with
**Qwen-Image-2512** at family defaults (50 steps, `true_cfg_scale` 4.0)
using same-seed prompt-variation — the cheap keyframe default per
ADR-033 §7 and `video-smoke/RESULTS.md` verdict 4.

| File | Seed | Role |
|---|---|---|
| `kf_000.jpg` | 4242 | seg 0 start — dog trotting toward camera, far down the misty path |
| `kf_001.jpg` | 4242 | seg 0 end / seg 1 start — dog reaches the moss-covered log |
| `kf_002.jpg` | 4242 | seg 1 end / seg 2 start — dog stopped, nose down on the moss |
| `kf_003.jpg` | 4242 | seg 2 end — close-up, head lifted, ears perked |
| `kf_100.jpg` | 8484 | seg 3 start — hilltop valley, empty sky (**scene cut**) |
| `kf_101.jpg` | 8484 | seg 3 end — same valley, birds crossing, sun higher |

Stored as quality-92 4:4:4 JPEG at full 1280×704 rather than PNG: the
source PNGs were ~1.3 MB each and tripped the repo's 500 KB large-file
pre-commit guard. Chroma subsampling is disabled, so the loss is
negligible for keyframe conditioning — and callers supply their own
keyframes in any PIL-readable format anyway.

`kf_001` and `kf_002` are each shared by two segments, so those joins are
*continuous*: the stitcher drops the duplicate boundary frame and
measures/applies color correction there. `kf_100` deliberately does not
match `kf_002`'s end, making segment 3 a **cut** — no dedup, no
correction, loud notice. That contrast is the point of the example.

Holding the seed within a scene is what keeps the forest layout, the log
position, and the valley ridgeline consistent across variations; the
scene-B pair (8484) differs only by birds and sun height.

Regenerate one with:

```
python -m comfyless.generate --model <hf-local>/Qwen-Image-2512 --prompt "<see git history>" --width 1280 --height 704 --seed 4242 --savepath kf_000
```
