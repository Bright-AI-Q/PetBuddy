# PetNet Project: Consolidated Bibliography Checklist

**Author:** PetBuddy Team  
**Date:** November 2025  
**Purpose:** This list integrates standard academic references with specific engineering resources found in the PetNet codebase. It is categorized logically to match the flow of the paper (Introduction $\to$ Related Work $\to$ Methodology $\to$ Experiments).

---

## 1. The Problem Context: Fine-Grained Visual Categorization (FGVC)
*Use these in the **Introduction** to explain why classifying pet breeds is difficult and why standard global feature extraction fails.*

*   **[1] Bilinear CNN Models for Fine-grained Visual Recognition**
    *   *Lin, T. Y., RoyChowdhury, A., & Maji, S. (ICCV 2015)*
    *   **Key Theory:** Uses second-order statistics to model part-feature interactions.
    *   **Role in Project:** Cited as the pioneering work in FGVC. We argue that while effective, Bilinear CNNs are computationally too heavy for mobile devices, justifying our lightweight **PetNet**.
    *   **Link:** [CVF PDF](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Lin_Bilinear_CNN_Models_ICCV_2015_paper.pdf)

*   **[2] Learning to Navigate for Fine-grained Classification (NTS-Net)**
    *   *Yang, Z., et al. (ECCV 2018)*
    *   **Key Theory:** Uses a "Navigator" agent to identify informative regions (like ears/eyes) and a "Teacher" to classify them.
    *   **Role in Project:** Represents the "Multi-Stage" approach. We contrast this with LDRE: "Instead of using a heavy region-proposal network like NTS-Net, we use efficient pre-computed masks (LDRE) to achieve similar focus on detailed parts."
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/1809.00287.pdf)

*   **[3] Destruction and Construction Learning for Fine-grained Image Recognition (DCL)**
    *   *Chen, Y., et al. (CVPR 2019)*
    *   **Key Theory:** Jigsaw-puzzle-style destruction forces the model to learn semantic relationships between parts.
    *   **Role in Project:** **Theoretical Support for LDRE**. We cite this to support the hypothesis that "destroying local structures (masking) forces the network to learn global robust features."
    *   **Link:** [CVF PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Destruction_and_Construction_Learning_for_Fine-Grained_Image_Recognition_CVPR_2019_paper.pdf)

---

## 2. The Foundation: Lightweight Architectures
*Use these in **Methodology** to describe our backbone choices and comparisons.*

*   **[4] MobileNetV2: Inverted Residuals and Linear Bottlenecks**
    *   *Sandler, M., et al. (CVPR 2018)*
    *   **Key Theory:** Inverted Residual Blocks (IRB) and Depthwise Separable Convolutions to reduce FLOPs.
    *   **Role in Project:** **The Backbone**. This is the foundation of PetNet. We explicitly state that we modify the standard MBV2 by adding attention mechanisms and distillation heads.
    *   **Link:** [CVF PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf)

*   **[5] EfficientNet: Rethinking Model Scaling for CNNs**
    *   *Tan, M., & Le, Q. (ICML 2019)*
    *   **Key Theory:** Compound scaling of depth, width, and resolution.
    *   **Role in Project:** **Comparison Target**. Used to represent "Heavyweight SOTA." We show that PetNet achieves comparable accuracy to larger models with a fraction of the parameters.
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/1905.11946.pdf)

---

## 3. Methodology A: Data Augmentation (Occlusion & Mixing)
*Use these in **Related Work** and **Methodology (LDRE)** to highlight our innovation.*

*   **[6] Random Erasing Data Augmentation**
    *   *Zhong, Z., et al. (AAAI 2020)*
    *   **Key Theory:** Randomly replacing rectangular regions with noise improves robustness.
    *   **Role in Project:** **The "Straw Man"**. We argue that Random Erasing is sub-optimal because it blindly erases background or useful features. LDRE is superior because it is *semantically guided*.
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/1708.04896.pdf)

*   **[7] GridMask Data Augmentation**
    *   *Chen, P., et al. (arXiv 2020)*
    *   **Key Theory:** Using a structured grid for occlusion avoids masking the entire object.
    *   **Role in Project:** **Precursor to LDRE**. We adopt the "grid" concept but refine it with "keypoint probability" (Anatomical Prior).
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/2001.04086.pdf)

*   **[8] mixup: Beyond Empirical Risk Minimization**
    *   *Zhang, H., et al. (ICLR 2018)*
    *   **Key Theory:** Linear interpolation of images and labels ($x' = \lambda x_i + (1 - \lambda)x_j$) smooths the decision boundary.
    *   **Role in Project:** **Synergistic Partner**. We explain that MixUp handles pixel-distribution shift, while LDRE handles structural overfitting. Together, they achieve SOTA.
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/1710.09412.pdf)

---

## 4. Methodology B: Architecture Modules (Attention & Distillation)
*Use these to explain the internal components of PetNet.*

*   **[9] ECA-Net: Efficient Channel Attention for Deep CNNs**
    *   *Wang, Q., et al. (CVPR 2020)*
    *   **Key Theory:** 1D convolution to determine channel importance without dimensionality reduction.
    *   **Role in Project:** The source of our **Dual-Attention** module. We chose it over SE-Net for its low parameter cost.
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/1910.03151.pdf)

*   **[10] Be Your Own Teacher (Self Distillation)**
    *   *Zhang, L., et al. (ICCV 2019)*
    *   **Key Theory:** Using the deepest layer of the network to supervise shallower layers (Auxiliary Heads).
    *   **Role in Project:** **Theoretical Basis for SelfKD**. We cite this to explain why we added auxiliary loss heads to Stage 1 and Stage 2 of PetNet.
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/1905.08094.pdf)

*   **[11] Attention Is All You Need**
    *   *Vaswani, A., et al. (NIPS 2017)*
    *   **Key Theory:** Self-Attention and Positional Encodings.
    *   **Role in Project:** Cited to explain the **2D Positional Encoding** component within our Dual-Attention module.
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/1706.03762.pdf)

---

## 5. Implementation: Domain-Specific Pose Datasets
*Use these in **Experiments** to explain how we trained the Keypoint Detector (YOLOv8-Pose) for LDRE.*

*   **[12] Who Left the Dogs Out? 3D Animal Reconstruction (StanfordExtra)**
    *   *Biggs, B., et al. (ECCV 2020)*
    *   **Role in Project:** **Crucial for LDRE**. Standard human-pose models fail on dogs. We cite this to explain how we fine-tuned YOLOv8 to accurately detect dog ears/noses for our masking algorithm.
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/2007.11110.pdf)

*   **[13] Animal Pose Dataset**
    *   *Cao, J., et al. (ICCV 2019)*
    *   **Role in Project:** **Crucial for LDRE**. Used to ensure our keypoint detector works for *Cats* as well as Dogs, ensuring the LDRE module is generic.
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/1908.10069.pdf)

*   **[14] AP-10K: A Benchmark for Animal Pose Estimation**
    *   *Yu, H., et al. (NeurIPS 2021)*
    *   **Role in Project:** General reference for animal pose estimation standards and pre-training data.
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/2108.12617.pdf)

---

## 6. Classification Benchmarks & Supplementary Data
*Use these in **Experiments** and **Data Preparation**.*

*   **[15] Cats and Dogs (Oxford-IIIT Pet Dataset)**
    *   *Parkhi, O. M., et al. (CVPR 2012)*
    *   **Role in Project:** Primary training/testing dataset.
    *   **Link:** [Official PDF](https://www.robots.ox.ac.uk/~vgg/publications/2012/Parkhi12a/parkhi12a.pdf)

*   **[16] Stanford Dogs Dataset**
    *   *Khosla, A., et al. (CVPR 2011 Workshop)*
    *   **Role in Project:** Primary fine-grained classification dataset.
    *   **Link:** [Official PDF](http://people.csail.mit.edu/khosla/papers/fgvc2011.pdf)

*   **[17] Microsoft COCO: Common Objects in Context**
    *   *Lin, T. Y., et al. (ECCV 2014)*
    *   **Role in Project:** **Foundational Data**. Used in `data_downloader.py` to fetch diverse background images or for pre-training.
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/1405.0312.pdf)

*   **[18] Public Domain Image Sources (Unsplash, Pexels, Pixabay)**
    *   **Role in Project:** **Robustness Testing**. We utilized custom scripts (`scarpe_cc0_api.py`) to harvest "Wild" test sets.

---

## 7. Tools & Visualization
*   **[19] Ultralytics YOLO**
    *   *Jocher, G., et al.*
    *   **Role in Project:** The engine used for object detection and keypoint extraction.
    *   **Project:** [GitHub](https://github.com/ultralytics/ultralytics)

*   **[20] Grad-CAM**
    *   *Selvaraju, R. R., et al. (ICCV 2017)*
    *   **Role in Project:** **Qualitative Analysis**. Used to visually prove that LDRE forces the model to look at the whole body.
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/1610.02391.pdf)

---

## 8. Theoretical Basis: Curriculum Learning & Strategy
*Use these in **Methodology (Training Strategy)** to justify the "MixUp First $\to$ LDRE Later" approach.*

*   **[21] Curriculum Learning**
    *   *Bengio, Y., et al. (ICML 2009)*
    *   **Key Theory:** Humans and machines learn better when examples are organized in meaningful order of difficulty.
    *   **Role in Project:** **Theoretical Foundation**. Justifies our "Training Strategy" (MixUp $\to$ LDRE).
    *   **Link:** [ACM PDF](https://dl.acm.org/doi/10.1145/1553374.1553380)

*   **[22] Curriculum by Smoothing**
    *   *Sinha, S., et al. (NeurIPS 2020)*
    *   **Key Theory:** Smoothing input data in early stages helps optimization.
    *   **Role in Project:** Explains why MixUp acts as an effective "starter" before the difficulty of LDRE.
    *   **Link:** [NeurIPS PDF](https://proceedings.neurips.cc/paper/2020/file/f6a4f494c259b3605c24e754593922c2-Paper.pdf)

---

## 9. Comparison Targets: Advanced Data Augmentation
*Use these in **Related Work** to contrast against LDRE.*

*   **[23] CutMix**
    *   *Yun, S., et al. (ICCV 2019)*
    *   **Key Theory:** Replacing a region with a patch from another image.
    *   **Role in Project:** **The "Straw Man"**. We compare LDRE against CutMix to show masking is better than pixel-mixing for fine-grained structure.
    *   **Link:** [CVF PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_with_Localizable_Features_ICCV_2019_paper.pdf)

*   **[24] SaliencyMix**
    *   *Uddin, A., et al. (CVPR 2021)*
    *   **Key Theory:** Selecting "salient" regions based on pixel intensity.
    *   **Role in Project:** **Direct Competitor**. We argue that our **Anatomical Saliency** (Keypoints) is superior to their pixel-based saliency.
    *   **Link:** [CVF PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Uddin_SaliencyMix_A_Saliency-Guided_Data_Augmentation_Method_for_Better_Regularization_CVPR_2021_paper.pdf)

---

## 10. Heavyweight & Modern Baselines (Transformers)
*Use these in **Experiments** to acknowledge the SOTA trend.*

*   **[25] TransFG: A Transformer Architecture for FGVC**
    *   *He, J., et al. (AAAI 2022)*
    *   **Role in Project:** **The "Heavyweight Champion"**. Cited to acknowledge the accuracy upper bound, but highlight its massive cost vs. PetNet.
    *   **Link:** [AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/view/19965)

*   **[26] MobileViT**
    *   *Mehta, S., & Rastegari, M. (ICLR 2021)*
    *   **Role in Project:** **Strong Baseline**. A modern lightweight competitor to compare against.
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/2110.02178.pdf)

---

## 11. 2024-2025 Cutting Edge: Future Context
*Use these to prove the paper is up-to-date.*

*   **[27] MobileNetV4: Universal Models for the Mobile Ecosystem**
    *   *Qin, D., et al. (CVPR 2024)*
    *   **Role in Project:** **Future Outlook**. We acknowledge V4 is SOTA, but validate V2 as the widest-supported standard for legacy edge devices.
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/2404.10518)

*   **[28] YOLOv10: Real-Time End-to-End Object Detection**
    *   *Wang, A., et al. (arXiv 2024)*
    *   **Role in Project:** **Module Update Path**. Our LDRE can seamlessly upgrade to YOLOv10 for lower latency.
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/2405.14458)

*   **[29] VMamba: Visual State Space Models**
    *   *Liu, Y., et al. (arXiv 2024)*
    *   **Role in Project:** **Defensive Citation**. We explain why we chose CNN over Mamba (Mamba lacks hardware support on low-end chips).
    *   **Link:** [arXiv PDF](https://arxiv.org/pdf/2401.10166)