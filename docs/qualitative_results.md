# 🖼️ Qualitative Results

This document presents a qualitative visual comparison of the segmentation masks generated across all experimental pipeline configurations. These visual results complement the quantitative performance metrics by illustrating how each network architecture handles fine-grained Multiple Sclerosis (MS) lesion boundary delineation and small focal lesion detection.

---

## 📌 Legend & Pipeline Nomenclature

| Subfigure | Pipeline |
| :---: | :--- |
| **(a)** | Input Image |
| **(b)** | Ground Truth |
| **(c)** | (P1) LR Baseline |
| **(d)** | HR Reference |
| **(e)** | P2: Frozen SR $\rightarrow$ Frozen Seg |
| **(f)** | P3: Frozen SR $\rightarrow$ Trainable Seg |
| **(g)** | P4: Trainable SR $\rightarrow$ Frozen Seg |
| **(h)** | P5: Joint E2E |
| **(i)** | P6: Joint Combined (Joint Loss) |
| **(j)** | P7: Sequential Joint (SR First) |
| **(k)** | P8: Sequential Joint (Seg First) |

---

## 🔬 Sample Case 1

<table align="center">
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_1/input_image.png" width="100%"/><br/><b>(a) Input Image</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_1/ground_truth.png" width="100%"/><br/><b>(b) Ground Truth</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_1/lr_segmentation_prediction.png" width="100%"/><br/><b>(c) LR Baseline</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_1/hr_segmentation_prediction.png" width="100%"/><br/><b>(d) HR Reference</b></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_1/frozen_sr-frozen_seg_prediction.png" width="100%"/><br/><b>(e) P2: F-SR / F-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_1/frozen_sr-trainable_seg_prediction.png" width="100%"/><br/><b>(f) P3: F-SR / T-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_1/trainable_sr-frozen_seg_prediction.png" width="100%"/><br/><b>(g) P4: T-SR / F-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_1/joint_e2e_prediction.png" width="100%"/><br/><b>(h) P5: Joint E2E</b></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_1/joint_combined_prediction.png" width="100%"/><br/><b>(i) P6: Joint Combined</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_1/sequential_joint_sr_first_prediction.png" width="100%"/><br/><b>(j) P7: Seq (SR First)</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_1/sequential_joint_seg_first_prediction.png" width="100%"/><br/><b>(k) P8: Seq (Seg First)</b></td>
    <td align="center" width="25%"></td>
  </tr>
</table>

## 🔬 Sample Case 2

<table align="center">
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_2/input_image.png" width="100%"/><br/><b>(a) Input Image</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_2/ground_truth.png" width="100%"/><br/><b>(b) Ground Truth</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_2/lr_segmentation_prediction.png" width="100%"/><br/><b>(c) LR Baseline</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_2/hr_segmentation_prediction.png" width="100%"/><br/><b>(d) HR Reference</b></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_2/frozen_sr-frozen_seg_prediction.png" width="100%"/><br/><b>(e) P2: F-SR / F-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_2/frozen_sr-trainable_seg_prediction.png" width="100%"/><br/><b>(f) P3: F-SR / T-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_2/trainable_sr-frozen_seg_prediction.png" width="100%"/><br/><b>(g) P4: T-SR / F-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_2/joint_e2e_prediction.png" width="100%"/><br/><b>(h) P5: Joint E2E</b></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_2/joint_combined_prediction.png" width="100%"/><br/><b>(i) P6: Joint Combined</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_2/sequential_joint_sr_first_prediction.png" width="100%"/><br/><b>(j) P7: Seq (SR First)</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_2/sequential_joint_seg_first_prediction.png" width="100%"/><br/><b>(k) P8: Seq (Seg First)</b></td>
    <td align="center" width="25%"></td>
  </tr>
</table>

## 🔬 Sample Case 3

<table align="center">
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_3/input_image.png" width="100%"/><br/><b>(a) Input Image</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_3/ground_truth.png" width="100%"/><br/><b>(b) Ground Truth</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_3/lr_segmentation_prediction.png" width="100%"/><br/><b>(c) LR Baseline</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_3/hr_segmentation_prediction.png" width="100%"/><br/><b>(d) HR Reference</b></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_3/frozen_sr-frozen_seg_prediction.png" width="100%"/><br/><b>(e) P2: F-SR / F-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_3/frozen_sr-trainable_seg_prediction.png" width="100%"/><br/><b>(f) P3: F-SR / T-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_3/trainable_sr-frozen_seg_prediction.png" width="100%"/><br/><b>(g) P4: T-SR / F-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_3/joint_e2e_prediction.png" width="100%"/><br/><b>(h) P5: Joint E2E</b></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_3/joint_combined_prediction.png" width="100%"/><br/><b>(i) P6: Joint Combined</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_3/sequential_joint_sr_first_prediction.png" width="100%"/><br/><b>(j) P7: Seq (SR First)</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_3/sequential_joint_seg_first_prediction.png" width="100%"/><br/><b>(k) P8: Seq (Seg First)</b></td>
    <td align="center" width="25%"></td>
  </tr>
</table>

## 🔬 Sample Case 4

<table align="center">
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_4/input_image.png" width="100%"/><br/><b>(a) Input Image</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_4/ground_truth.png" width="100%"/><br/><b>(b) Ground Truth</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_4/lr_segmentation_prediction.png" width="100%"/><br/><b>(c) LR Baseline</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_4/hr_segmentation_prediction.png" width="100%"/><br/><b>(d) HR Reference</b></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_4/frozen_sr-frozen_seg_prediction.png" width="100%"/><br/><b>(e) P2: F-SR / F-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_4/frozen_sr-trainable_seg_prediction.png" width="100%"/><br/><b>(f) P3: F-SR / T-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_4/trainable_sr-frozen_seg_prediction.png" width="100%"/><br/><b>(g) P4: T-SR / F-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_4/joint_e2e_prediction.png" width="100%"/><br/><b>(h) P5: Joint E2E</b></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_4/joint_combined_prediction.png" width="100%"/><br/><b>(i) P6: Joint Combined</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_4/sequential_joint_sr_first_prediction.png" width="100%"/><br/><b>(j) P7: Seq (SR First)</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_4/sequential_joint_seg_first_prediction.png" width="100%"/><br/><b>(k) P8: Seq (Seg First)</b></td>
    <td align="center" width="25%"></td>
  </tr>
</table>

## 🔬 Sample Case 5

<table align="center">
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_5/input_image.png" width="100%"/><br/><b>(a) Input Image</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_5/ground_truth.png" width="100%"/><br/><b>(b) Ground Truth</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_5/lr_segmentation_prediction.png" width="100%"/><br/><b>(c) LR Baseline</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_5/hr_segmentation_prediction.png" width="100%"/><br/><b>(d) HR Reference</b></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_5/frozen_sr-frozen_seg_prediction.png" width="100%"/><br/><b>(e) P2: F-SR / F-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_5/frozen_sr-trainable_seg_prediction.png" width="100%"/><br/><b>(f) P3: F-SR / T-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_5/trainable_sr-frozen_seg_prediction.png" width="100%"/><br/><b>(g) P4: T-SR / F-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_5/joint_e2e_prediction.png" width="100%"/><br/><b>(h) P5: Joint E2E</b></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_5/joint_combined_prediction.png" width="100%"/><br/><b>(i) P6: Joint Combined</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_5/sequential_joint_sr_first_prediction.png" width="100%"/><br/><b>(j) P7: Seq (SR First)</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_5/sequential_joint_seg_first_prediction.png" width="100%"/><br/><b>(k) P8: Seq (Seg First)</b></td>
    <td align="center" width="25%"></td>
  </tr>
</table>

## 🔬 Sample Case 6

<table align="center">
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_6/input_image.png" width="100%"/><br/><b>(a) Input Image</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_6/ground_truth.png" width="100%"/><br/><b>(b) Ground Truth</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_6/lr_segmentation_prediction.png" width="100%"/><br/><b>(c) LR Baseline</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_6/hr_segmentation_prediction.png" width="100%"/><br/><b>(d) HR Reference</b></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_6/frozen_sr-frozen_seg_prediction.png" width="100%"/><br/><b>(e) P2: F-SR / F-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_6/frozen_sr-trainable_seg_prediction.png" width="100%"/><br/><b>(f) P3: F-SR / T-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_6/trainable_sr-frozen_seg_prediction.png" width="100%"/><br/><b>(g) P4: T-SR / F-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_6/joint_e2e_prediction.png" width="100%"/><br/><b>(h) P5: Joint E2E</b></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_6/joint_combined_prediction.png" width="100%"/><br/><b>(i) P6: Joint Combined</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_6/sequential_joint_sr_first_prediction.png" width="100%"/><br/><b>(j) P7: Seq (SR First)</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_6/sequential_joint_seg_first_prediction.png" width="100%"/><br/><b>(k) P8: Seq (Seg First)</b></td>
    <td align="center" width="25%"></td>
  </tr>
</table>

## 🔬 Sample Case 7

<table align="center">
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_7/input_image.png" width="100%"/><br/><b>(a) Input Image</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_7/ground_truth.png" width="100%"/><br/><b>(b) Ground Truth</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_7/lr_segmentation_prediction.png" width="100%"/><br/><b>(c) LR Baseline</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_7/hr_segmentation_prediction.png" width="100%"/><br/><b>(d) HR Reference</b></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_7/frozen_sr-frozen_seg_prediction.png" width="100%"/><br/><b>(e) P2: F-SR / F-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_7/frozen_sr-trainable_seg_prediction.png" width="100%"/><br/><b>(f) P3: F-SR / T-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_7/trainable_sr-frozen_seg_prediction.png" width="100%"/><br/><b>(g) P4: T-SR / F-Seg</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_7/joint_e2e_prediction.png" width="100%"/><br/><b>(h) P5: Joint E2E</b></td>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_7/joint_combined_prediction.png" width="100%"/><br/><b>(i) P6: Joint Combined</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_7/sequential_joint_sr_first_prediction.png" width="100%"/><br/><b>(j) P7: Seq (SR First)</b></td>
    <td align="center" width="25%"><img src="../.repo/qualitative_results/batch_7/sequential_joint_seg_first_prediction.png" width="100%"/><br/><b>(k) P8: Seq (Seg First)</b></td>
    <td align="center" width="25%"></td>
  </tr>
</table>