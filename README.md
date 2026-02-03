# CYGD: Calabi–Yau–Inspired Geometric Descriptors

This repository implements **Calabi–Yau–inspired geometric descriptors (CYGD)** for protein structure analysis, as described in the manuscript *“Calabi–Yau–Inspired Geometric Descriptors for Protein Structure Analysis”*.

CYGD characterizes protein structures by **quantifying structured deviations from idealized geometric organization inspired by Calabi–Yau geometry**, rather than assuming proteins satisfy Calabi–Yau conditions in a strict mathematical sense.

The code is organized into two main folders corresponding to two complementary geometric viewpoints.

---

## CY-A (Angle-based CYGD)

The **CY-A** folder contains all functions and pipelines required to construct **angle-based Calabi–Yau–inspired geometric descriptors** from protein torsion angles.

Taken together, the code in this folder:
- extracts backbone and side-chain torsion angles from protein structures,
- embeds angular variables into a continuous geometric space,
- constructs local residue neighborhoods at multiple scales,
- quantifies deviations from Lagrangian and special Lagrangian–inspired geometric conditions,
- produces multi-scale geometric descriptors that can be used directly for residue-level prediction or aggregated into global representations,
- supports CY-like ring visualizations and associated global scores for qualitative geometric intuition.



---

## CY-C (Coordinate-based CYGD)

The **CY-C** folder contains all functions and pipelines required to construct **coordinate-based Calabi–Yau–inspired geometric descriptors** directly from three-dimensional atomic coordinates.

Taken together, the code in this folder:
- extracts backbone atomic coordinates from protein structures,
- constructs local residue neighborhoods at multiple scales,
- evaluates geometric compatibility inspired by complex, Kähler, and Calabi–Yau structures,
- quantifies deviations in local coordinate geometry and volumetric behavior,
- produces multi-scale geometric descriptors that can be used directly or aggregated into global representations,
- supports CY-like ring visualizations and associated global scores for qualitative geometric intuition.

---

## Code and data availability


The datasets used in this study are publicly available and are described in detail in the main text.

If you need the  data used in the manuscript, please request it by contacting **huguoqing@himis-sz.cn**.

