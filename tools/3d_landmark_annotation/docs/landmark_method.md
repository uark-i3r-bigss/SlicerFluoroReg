# Landmark Method

How the pipeline gets from a CT volume to the landmark FCSV, and what each landmark key means.

## Segmentation and cleanup

1. TotalSegmentator generates individual masks.
2. Masks used by the landmark workflow are cleaned with largest-connected-component filtering to
   remove small isolated islands that corrupt STL generation, ROI boxes, and landmarks.
3. The selected masks are converted into STL meshes for geometry-based analysis.

Pipeline-generated meshes and final landmark FCSVs use **LPS world coordinates**. NIfTI vertices are
transformed through the image affine and then converted from nibabel RAS+ world coordinates by
negating `x` and `y`. The anatomy-local frames below are used only during detection.

## Pelvis

Pelvis landmarks are extracted from the combined hip mesh plus the left and right hip meshes.
Detection uses a right-handed pelvis-local frame:

- origin: midpoint of the left and right unique-vertex hip-mesh centroids (`PEL_COM`), not a volume-
  or mass-weighted pelvis centroid
- `x`: from the right hip centroid toward the left hip centroid
- `z`: dominant pelvis direction after removing the left-right component, with its sign selected from
  pelvis-end spread and inferomedial shape
- `y`: `z × x`; the landmark searches treat `-y` as anterior and `+y` as posterior

Landmarks:

- `PEL_ASIS_L/R`: anterior iliac tip candidates on the anterior-lateral crest
- `PEL_AIIS_L/R`: anterior inferior iliac spine candidates inferior to the ASIS landmarks
- `PEL_PT_L/R`: pubic tubercle candidates from the inferior medial pelvis
- `PEL_IT_L/R`: ischial tuberosity candidates from the inferior posterior pelvis
- `PEL_IPR_L/R`: unvalidated inferior pubic ramus candidates; computed internally but suppressed from
  the final FCSV
- `PEL_PS`: pubic symphysis estimate, the midpoint of the pubic tubercle candidates
- `PEL_PSIS_L/R`: posterior superior iliac spine candidates calibrated from the pelvis-local ASIS
  relationship
- `PEL_FH_L/R`: pelvis-side femoral head centers (the left and right acetabular socket centers),
  copied verbatim from `FEM_L_FH` and `FEM_R_FH`

`PEL_FH_L/R` and `FEM_L_FH`/`FEM_R_FH` are intentionally the same physical point expressed in two
object namespaces: the acetabular socket center coincides with the femoral head center. The pelvis
copies exist so pelvis-only pose estimation has well-separated geometry instead of nearly coincident
pubic and iliac points. They are not redundant duplicates and must never be de-duplicated or given
different values. The pelvis points are not recomputed: they are written only when the corresponding
femoral head center was fitted, with identical LPS coordinates. If a femur is missing or its sphere
fit fails, that side's pelvis point is omitted rather than substituted.

After ASIS and pubic landmarks are available, the pipeline records a right-handed anterior pelvic
plane (APP) debug frame: origin `PEL_PS`; `x` from right ASIS to left ASIS; `y` the APP normal
aligned with pelvis-local `+y`; `z` = `x × y`. The APP frame is diagnostic output and does not
transform the stored landmarks.

## Femur

Femur detection uses two axes. The head search keeps the whole-mesh PCA direction, signed so the
endpoint with greater transverse spread is proximal. That direction leans toward the neck by an
amount that depends on how much shaft the scan holds: a 441 mm femur cut to 140-180 mm tilts it
11-18 degrees. The trochanter landmarks are extrema in the femur frame, so their frame instead uses
the diaphysis axis: the line through circle centres of 3 mm slices 100-150 mm below the femur's
most proximal point (just distal to the lesser trochanter), refined three times. Every femur
longer than about 160 mm holds that window in full, so a CBCT-length femur gets the same frame as
the full bone. After sphere fitting locates the femoral head, the final right-handed frame is:

- origin: fitted femoral head center
- `z`: the diaphysis axis, pointing distal-to-proximal
- `x`: projected direction from the proximal shaft center toward the femoral head, generally medial
- `y`: `z × x`

When the mesh ends less than 30 mm into the window (a femur shorter than about 140 mm), the frame
falls back to the PCA axis and `FT_S`, `FT_T` and `LT` are written as `low_confidence` with reason
`frame_unresolved`. The debug JSON records `axis_source`, `diaphysis_window_mm` and
`pca_axis_angle_deg` for each femur frame.

Because `x` points medially on both sides, the anatomical direction of `y` is mirrored between left
and right femurs. The frame is mesh-derived and does not use the pelvis APP.

Landmarks:

- `FEM_L_FH`, `FEM_R_FH`: femoral head centers from sphere fitting in the proximal femur; the same
  coordinates are also emitted in the pelvis namespace as `PEL_FH_L/R`. A fit is accepted only if
  its radius is within 12-40 mm, its inlier RMS is at most 3 mm, its inliers back at least half of
  the sphere's directions (`sphere_support`) and span at least a cap of 53-degree half-angle, and
  its bootstrap centre spread is at most 1 mm; otherwise the head abstains and neither namespace
  gets the point. The inlier fraction of the isolated head region is reported but not used: that
  region also holds neck and medial shaft surface.
- `FEM_L_FT_S`, `FEM_R_FT_S`: greater trochanter superior landmarks
- `FEM_L_FT_T`, `FEM_R_FT_T`: greater trochanter lateral tip landmarks
- `FEM_L_LT`, `FEM_R_LT`: lesser trochanter landmarks from the strongest residual prominence among
  medial proximal candidates

## Lumbar and sacrum

Lumbar landmarks combine whole-vertebra mesh centroids with scanner-axis extrema:

- `LUM_L1_COM` … `LUM_L5_COM`: mean of the unique vertices of each whole-vertebra mesh, not an
  isolated vertebral-body or volume/mass-weighted center
- `LUM_L1_SP` … `LUM_L5_SP`: spinous process landmarks
- `LUM_L1_TP_L/R` … `LUM_L5_TP_L/R`: transverse process landmarks
- `LUM_SAC_A`: sacral apex

## Landmark keys

Canonical keys, output labels, visibility, and descriptions live in
[`configs/landmark_config.json`](../configs/landmark_config.json). Default output labels are
identical to the internal keys.

| Region | Landmark | Key |
| --- | --- | --- |
| Pelvis | Left anterior superior iliac spine | `PEL_ASIS_L` |
| Pelvis | Right anterior superior iliac spine | `PEL_ASIS_R` |
| Pelvis | Left anterior inferior iliac spine | `PEL_AIIS_L` |
| Pelvis | Right anterior inferior iliac spine | `PEL_AIIS_R` |
| Pelvis | Left pubic tubercle | `PEL_PT_L` |
| Pelvis | Right pubic tubercle | `PEL_PT_R` |
| Pelvis | Left ischial tuberosity | `PEL_IT_L` |
| Pelvis | Right ischial tuberosity | `PEL_IT_R` |
| Pelvis | Left inferior pubic ramus candidate; not exported | `PEL_IPR_L` |
| Pelvis | Right inferior pubic ramus candidate; not exported | `PEL_IPR_R` |
| Pelvis | Pubic symphysis estimate from pubic tubercle midpoint | `PEL_PS` |
| Pelvis | Midpoint of left/right hip-mesh centroids | `PEL_COM` |
| Pelvis | Left posterior superior iliac spine | `PEL_PSIS_L` |
| Pelvis | Right posterior superior iliac spine | `PEL_PSIS_R` |
| Pelvis | Left acetabular/femoral head center; same point as `FEM_L_FH` | `PEL_FH_L` |
| Pelvis | Right acetabular/femoral head center; same point as `FEM_R_FH` | `PEL_FH_R` |
| Femur | Left femoral head center; same point as `PEL_FH_L` | `FEM_L_FH` |
| Femur | Right femoral head center; same point as `PEL_FH_R` | `FEM_R_FH` |
| Femur | Left greater trochanter superior | `FEM_L_FT_S` |
| Femur | Right greater trochanter superior | `FEM_R_FT_S` |
| Femur | Left greater trochanter lateral tip | `FEM_L_FT_T` |
| Femur | Right greater trochanter lateral tip | `FEM_R_FT_T` |
| Femur | Left lesser trochanter | `FEM_L_LT` |
| Femur | Right lesser trochanter | `FEM_R_LT` |
| Lumbar | L1–L5 whole-vertebra mesh centroid | `LUM_L<n>_COM` |
| Lumbar | L1–L5 spinous process | `LUM_L<n>_SP` |
| Lumbar | L1–L5 left/right transverse process | `LUM_L<n>_TP_L` / `LUM_L<n>_TP_R` |
| Sacrum | Sacral apex | `LUM_SAC_A` |

## Configuration

- [`configs/landmark_config.json`](../configs/landmark_config.json): canonical keys, output labels,
  visibility, descriptions.
- [`configs/lumbar_pelvis_crop_config.json`](../configs/lumbar_pelvis_crop_config.json): crop groups,
  margins, and which landmarks are written into each crop folder.
- [`configs/totalseg_all_labels.json`](../configs/totalseg_all_labels.json): combined full-label
  export after TotalSegmentator.

The `label` field in `landmark_config.json` may be changed for **standalone-only exports**. Do not
rename labels or add aliases for integrated registration: downstream consumers look up the native
names. `visible: false` suppresses a landmark row entirely; `DBG_*` keys are always excluded from the
final FCSV. Do not re-enable unvalidated landmarks by changing visibility or crop membership.

## Known limitations

- Pelvis and femur orientation is inferred from mesh geometry rather than scanner-axis extrema;
  incorrect or incomplete segmentations can still reverse or destabilize an inferred axis.
- Femur frames are side-local: their medial `x` axes make their `y` directions anatomically mirrored.
- Lumbar process landmarks remain scanner-axis extrema; `COM` values are mesh centroids as above.
- `PEL_IPR_L/R` remain unvalidated and are omitted from final FCSV and crop outputs.
- Successful job completion is not evidence of anatomical accuracy. Real-case review is still
  required for pelvis/femur orientation, landmark placement, and partial-field-of-view behavior.
