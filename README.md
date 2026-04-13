# Urine Culture based Urinary Tract Infection (UTI) detection and classification using Deep Learning

Urine culture is a critical diagnostic methodology for identifying bacterial infections through microscopic observation in clinical settings. The traditional approach involves expert examination of Petri dishes for urine culture, which is both labor-intensive and time-consuming. While machine intelligence cannot replace specialist supervision, it can complement expert oversight using machine-based pattern recognition.

Urine samples corresponding to cultures with urinary tract infection (UTI) susceptibility exhibit certain patterns that can be effectively recognized by deep learning techniques. This research proposes employing deep learning techniques to identify bacterial infections in urine cultures.

Data Availability Statement: The dataset of interest in this article can be accessed from the article,
https://www.sciencedirect.com/science/article/pii/S235234092300152X.

Traditionally, multi-layered perceptrons (`MLP`) have contributed to the architecture of numerous deep-learning models, such as Convolutional Neural Networks (CNNs) and Transformers, which are known to complement labor-intensive tasks with high accuracy. Recently, a new architecture for deep-learning models, based on the **Kolmogorov–Arnold Representation Theorem**, has been proposed. Models adopting the **Kolmogorov Arnold Network** (`KAN`) architecture have demonstrated better results in many instances compared to those based on classical `MLP` architectures.

Experiments conducted in this research used a dataset of 1,500 urine culture images contributed by da Silva *et al.*, annotated into three categories: **positive**, **negative**, and **uncertain** UTI susceptibility. The results showed that models incorporating the `KAN` architecture achieved superior accuracy compared to their `MLP`-based counterparts. For instance:

- **`KAN-C-Norm`**: 86.95 ± 0.56%
- **`KAN-C-MLP`**: 87.16 ± 0.97%

In contrast, models based on `MLP` architectures demonstrated lower performance:

- **Vision Transformer (`ViT`)**: 80.33 ± 0.92%
- **Class-Attention in Vision Transformers (`CAiT`)**: 78.66 ± 2.63%

These findings highlight the potential for machine intelligence to assist medical experts in improving the UTI diagnosis process. By leveraging advanced deep learning architectures like `KAN`, the accuracy and efficiency of diagnosing bacterial infections in urine cultures can be significantly enhanced.
