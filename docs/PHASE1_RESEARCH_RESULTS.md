# Phase 1 Research Results & Technical Report

**Project:** UtiTherm-SR  
**Author:** Syed Huzaifa Chishty (SMHC)  
**Evaluation Protocol:** Step 11L — Predeclared Sequence-Disjoint FLIR Video Test (`STEP11L_PREDECLARED_SEQUENCE_DISJOINT_FLIR_VIDEO_TEST_V1`)  
**Status:** Completed Phase 1  
**Reproducibility Verification:** Bit-for-bit identical predictions across independent container executions (`step11l_full_3749_a` and `step11l_full_3749_b`)

---

## 1. Executive Summary

UtiTherm-SR investigates whether selectively applying neural super-resolution (SR) to image regions containing candidate objects improves downstream thermal object detection utility compared to uniform full-image Bicubic interpolation or full-image neural SR under fixed computational budgets.

Evaluating 3,749 sequence-disjoint thermal images across eight held-out video sequences from the FLIR ADAS v2 dataset family revealed:
1. **Full Bicubic interpolation achieved the highest detection utility** ($24,220.50$) and the lowest per-frame inference latency ($10.31\text{ ms}$).
2. **Selective neural super-resolution did not improve utility**: Contrast-based top-4 region enhancement achieved utility $24,091.25$ ($19.31\text{ ms}$), performing worse than Full Bicubic due to an increase of $+477$ false positives while recovering $10$ fewer small-object true positives.
3. **Full-frame Large FSRCNN performed worst among all evaluated methods**: Despite sharpening edges, full neural SR produced utility $23,542.00$ ($32.74\text{ ms}$), suffering a drop of $-730$ small true positives compared to Bicubic.
4. **Development-observed threshold gains failed to generalize**: A confidence threshold shift from $0.25$ to $0.40$ appeared beneficial on development frames, but on unseen test sequences Bicubic retained superiority over all neural variants at both thresholds.

---

## 2. Research Questions

- **RQ1 (Detection Utility vs Interpolation):** Does applying neural super-resolution (FSRCNN) to thermal images yield higher downstream object detection true positives and fewer false positives than simple Bicubic interpolation?
- **RQ2 (Selective vs Full-Frame Allocation):** Can region-of-interest (ROI) selection algorithms (contrast ranking, learned utility regression, random control) allocate compute to enhance only critical areas without degrading background context?
- **RQ3 (Latency vs Accuracy Trade-off):** Does partial neural enhancement offer a superior Pareto trade-off between latency (ms) and small-object detection accuracy?
- **RQ4 (Generalization across Video Sequences):** Do hyperparameter choices (such as detection confidence thresholds) tuned on development splits hold up when evaluated on unseen video sequences?

---

## 3. System Architecture & Methods

The UtiTherm-SR evaluation pipeline operates on 8-bit thermal low-resolution images ($320 \times 256$):

```
Thermal LR Input (320x256)
       │
       ├──► Base Path: Full-frame Bicubic Upsampling ────► Canvas (640x512)
       │                                                         ▲
       └──► Selective Path:                                     │
              │                                                  │
              ├──► Grid Partitioning (4x4 tiles, 160x128 HR)     │
              │                                                  │
              ├──► Region Ranking (Thermal Contrast / Variance) │
              │                                                  │
              ├──► Top-k Neural SR (Large FSRCNN x2)             │
              │                                                  │
              └──► Feathered Blending (8px alpha border) ────────┘
                               │
                               ▼
                   Composited Frame (640x512)
                               │
                               ▼
                   YOLO11n Object Detector
                               │
                               ▼
               Detection Utility & Latency Metrics
```

### Evaluated Methods
1. **Full Bicubic (`full_bicubic_640`):** Full-frame Bicubic interpolation to $640 \times 512$. Pure interpolation baseline with zero neural network overhead during enhancement.
2. **Full Large FSRCNN (`full_large_fsrcnn_640`):** Full-frame neural super-resolution using an expanded FSRCNN architecture ($d=64, s=16, m=6$, $2\times$ scale) trained on FLIR thermal image pairs.
3. **Random 4-Region Enhancement (`random4_bicubic_first_h10`):** Immutable Bicubic $640 \times 512$ base image; 4 tiles chosen uniformly at random (fixed seed) enhanced with Large FSRCNN and blended using 8px feathering.
4. **Contrast 4-Region Enhancement (`contrast4_bicubic_first_h10`):** Immutable Bicubic $640 \times 512$ base image; candidate tiles ranked by local thermal contrast (pixel variance in LR); top-4 tiles enhanced with Large FSRCNN and blended with 8px feathering.

---

## 4. Evaluation Protocol & Provenance

### Metric Definition
Downstream object detection performance is evaluated using a bespoke **Detection Utility** score designed for safety-critical thermal perception:
$$\text{Utility} = \text{Small TP} - 0.25 \times \text{FP}$$

- **Small TP:** True positive detections on small objects ($\text{area} < 32^2$ pixels in $640 \times 512$ coordinates) with class-aware $\text{IoU} \ge 0.50$.
- **FP:** All false positive detections across all supported classes (person, car, bicycle, etc.).
- **Latency:** Deterministic per-frame median GPU execution time in milliseconds (excluding disk I/O).

*Note: This diagnostic metric penalizes false alarms against small-object detection. It is explicitly not standard COCO mAP.*

### Dataset & Split Integrity
- **Source:** FLIR ADAS v2 thermal 8-bit dataset family.
- **Split Separation:**
  - Training: Used for detector and SR training.
  - Development Holdout: 200 images used during algorithm development (Steps 7–11G).
  - Test Split (Step 11K/11L): **3,749 images** strictly partitioned across **8 independent video sequences**, with zero sequence overlap with training or development data.
  - Leakage & Preprocessing Audit: Verified in Step 11J (1,144/1,144 validation annotation conversions and 51/51 exact PIL Bicubic preprocessing matches).

### Strict Determinism
All evaluations enforce locked PyTorch determinism:
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `torch.use_deterministic_algorithms(True)`
- `torch.backends.cudnn.benchmark = False`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.allow_tf32 = False`
- `torch.backends.cuda.matmul.allow_tf32 = False`
- `cv2.setNumThreads(1)`

---

## 5. Experimental Results

### Primary Comparison (Confidence Threshold = 0.25)
Evaluated on **3,749 test images** across **8 video sequences**, 3 repeats per frame:

| Method | Small TP | Small GT | Small Recall | FP | Absolute Utility | Delta Utility vs Bicubic | Mean GPU Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Full Bicubic Baseline** | **28,401** | 58,379 | **48.65%** | 16,722 | **24,220.50** | *Reference* | **10.31 ms** |
| **Random 4 Regions** | 28,302 | 58,379 | 48.48% | 16,825 | 24,095.75 | -124.75 | 18.94 ms |
| **Contrast 4 Regions** | 28,391 | 58,379 | 48.63% | 17,199 | 24,091.25 | -129.25 | 19.31 ms |
| **Full Large FSRCNN** | 27,671 | 58,379 | 47.40% | **16,516** | 23,542.00 | -678.50 | 32.74 ms |

#### Primary Hypothesis Test: Contrast-4 vs Full Bicubic (Conf = 0.25)
- $\Delta \text{Small TP} = -10$
- $\Delta \text{FP} = +477$
- $\Delta \text{Utility Total} = -129.25$ ($\Delta \text{Utility/image} = -0.034$)
- Per-image wins / ties / losses: 1,280 wins, 938 ties, 1,531 losses (Bicubic won 54.4% of non-tied frames)
- Sequence-level breakdown: Full Bicubic outperformed Contrast-4 in **6 out of 8** independent video sequences.
- 95% Video-Cluster Bootstrap CI: $[-0.242, +0.177]$ per frame.

---

### Secondary Confidence Analysis (Confidence Threshold = 0.40)

During earlier development on a 200-frame validation subset (Step 11F/11G/11I), raising the detector confidence cutoff to $0.40$ appeared to suppress false positives and showed apparent utility gains for Contrast-4.

However, when tested on the locked, sequence-disjoint test set:

| Method | Small TP (conf 0.40) | FP (conf 0.40) | Absolute Utility (conf 0.40) | Delta vs Bicubic @ 0.40 | Delta vs Bicubic @ 0.25 |
|---|---:|---:|---:|---:|---:|
| **Full Bicubic** | 23,763 | **5,681** | **22,342.75** | *Reference* | -1,877.75 |
| **Random 4 Regions** | 23,662 | 5,986 | 22,165.50 | -177.25 | -2,055.00 |
| **Contrast 4 Regions** | **23,767** | 6,427 | 22,160.25 | -182.50 | -2,060.25 |
| **Full Large FSRCNN** | 23,371 | 6,503 | 21,745.25 | -597.50 | -2,475.25 |

#### Findings at Confidence 0.40:
1. **Development observations did not generalize:** On unseen video sequences, Contrast-4 produced only $+4$ additional small TPs ($23,767$ vs $23,763$) but introduced $+746$ additional false positives ($6,427$ vs $5,681$), leading to a net utility loss of $-182.50$.
2. **Operating Point Optimality:** Operating at confidence $0.25$ yielded substantially higher overall utility ($24,220.50$) than operating at $0.40$ ($22,342.75$) because the small-object true positive drop was severe ($-4,638$ small TPs).
3. **Sequence wins:** Full Bicubic defeated Contrast-4 in **6 of 8** video sequences at confidence $0.40$.

---

## 6. Spatial False-Positive Analysis

An audit of detection errors (Step 11H) investigated where and why neural enhancement creates false positives:
- **High-gradient thermal boundaries:** Neural SR sharpens edges of heat plumes, building roofs, and vehicle exhaust pipes, creating contrast patterns that mimic distant pedestrians or cyclists to YOLO11n.
- **Tile Boundary Transitions:** Even with 8-pixel feathered alpha blending, subtle tonal transitions between neural patches and Bicubic backgrounds trigger detector edge activations.
- **Detector Misalignment:** YOLO11n fine-tuned on Bicubic thermal images is sensitive to high-frequency artifacts generated by FSRCNN transposed convolutions.

---

## 7. Deterministic Reproducibility Verification

Two independent end-to-end evaluation runs were executed across separate Docker containers on an NVIDIA GeForce RTX 5070:
- **Run A:** `reports/test/step11l_full_3749_a/summary.json`
- **Run B:** `reports/test/step11l_full_3749_b/summary.json`

### Verification Metrics:
- **Total frames evaluated:** 3,749 images $\times$ 4 methods $\times$ 3 repeats = 44,988 inferences per run.
- **Prediction agreement:** Exactly **0 mismatches** across all 3,749 frames (`cross_run_mismatch_count: 0`).
- **Fingerprint verification:** All rendered uint8 image SHA256 hashes, exact floating-point YOLO box coordinates, confidence scores, and class labels matched identically.
- **Provenance hashes:**
  - Config SHA256: `935a1c68ed180ed43ef40a4081e357f6a797331edc585f8ac8e2a3ef4c56ee1e`
  - Prepared image names SHA256: `ae03a2bed443ce4df798946be1941e7c852148a3cad92a407d3a1529165cfc33`
  - Detector weights SHA256: `890ed93b8d6466244ebe703c0955414b535e37672db19dce08da6551b816677c`
  - Large FSRCNN weights SHA256: `445b4adc8ad53c18274cf7d101ca54ac1e5ab7bea35c85bd53b5688ff729a5ae`

---

## 8. Threats to Validity & Limitations

1. **Dataset Family:** All evaluation was conducted on FLIR ADAS v2. Cross-camera and cross-sensor generalization (e.g., to Boson or uncooled microbolometers with different NUC characteristics) was not evaluated.
2. **Limited Independent Sequences:** Although 3,749 frames were evaluated, they originate from 8 continuous driving video sequences. Individual sequential frames exhibit high temporal autocorrelation.
3. **Metric Sensitivity:** The bespoke utility metric depends on the chosen false-positive penalty ($\lambda = 0.25$). Different operational cost matrices would alter the absolute numerical trade-offs.
4. **Synthetic Degradation:** FSRCNN was trained on synthetically downsampled Bicubic pairs; real optical lens blurring and sensor modulation transfer function (MTF) characteristics were not modeled.

---

## 9. Future Research: Phase 2 (FideTherm-SR)

To address the limitations uncovered in Phase 1, follow-up research is proposed under the project name **FideTherm-SR** (*Fidelity-Preserving and Detection-Aware Thermal Super-Resolution*):
- **Detection-Aware Training Loss:** Training the super-resolution network with downstream object detection gradient feedback (feature loss / perceptual loss from detector backbones) rather than pixel MSE alone.
- **Physical Thermal Degradation Modeling:** Incorporating non-uniformity correction (NUC) noise, thermal diffusion, and optical point spread function (PSF) degradations.
- **Adaptive Region Budgets:** Moving beyond fixed grid tiles to bounding-box-guided or saliency-driven non-rectangular continuous blending.
- **Cross-Dataset Validation:** Benchmarking on distinct thermal sensor datasets (KAIST Multispectral, Teledyne FLIR Thermal Dataset, etc.).

*(Note: FideTherm-SR is a proposed future research direction; Phase 1 remains the completed foundation.)*
