# Green Federated Learning: A Data-Centric Approach in the Cloud Continuum

---

## Table of Contents
- [Introduction](#introduction)
- [State of the Art](#state-of-the-art)
- [Goals and Requirements](#goals-and-requirements)
- [Green FL Methodology](#green-fl-methodology)
- [Implementation](#implementation)
  - [Data-Centric FL Exploration](#data-centric-fl-exploration)
  - [Data Analysis and Modelling](#data-analysis-and-modelling)
  - [FL Configuration Recommendation](#fl-configuration-recommendation)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [References](#references)

---

## Introduction
The dynamic landscape of Artificial Intelligence (AI) and Machine Learning (ML) has gained widespread attention combined with a world dominated by IoT devices and concepts such as cloud, edge, and Fog Computing. Furthermore, the demand for computational efficiency and sustainability has become paramount. The environmental challenges posed by Deep Learning applications necessitate a shift from Red AI to Green AI [^5], emphasizing efficiency in creating a sustainable, data-driven world. Within this context, data quality issues and the need for privacy are dominant. The research project aims to address these challenges by exploring the Federated Learning (FL) approach. The work focuses on developing a data-centric methodology to optimize data usage, reducing carbon emissions and enhancing FL system efficiency.

## State of the Art
### Key Challenges:
- **High AI Training Costs**: ML models require vast amounts of energy.
- **Carbon Footprint of AI**: Deep learning contributes significantly to carbon emissions.
- **Data Quality and Privacy Issues**: Ensuring quality while maintaining privacy is crucial.

### Related Research:
- Traditional centralized approaches focus on reducing emissions by optimizing dataset size.
- Federated Learning mitigates privacy concerns but introduces **heterogeneity challenges**.
- Fog Computing provides a **distributed** model for energy-efficient FL training.

## Goals and Requirements
### Goals:
1. **Analyze data-centric impact on FL energy consumption**: Assess the effects of data volume and quality on accuracy and emissions.
2. **Minimize carbon emissions in FL**: Develop an optimized node and data selection methodology.

### Requirements:
- **R1**: Reduce emissions in FL training.
- **R2**: Maintain model performance.
- **R3**: Provide a generalizable approach for FL configuration.

## Green FL Methodology
### Overview:
The **FL Configuration Selection System** provides recommendations to optimize FL training with minimal environmental impact. It operates in three phases:
1. **Data-Centric FL Exploration**
2. **Data Analysis and Modelling**
3. **FL Configuration Recommendation**

*(Insert Image: FL Configuration Selection System Architecture)*

## Implementation
### Data-Centric FL Exploration
- Uses the **Flower** FL framework.
- Simulations assess the impact of **data volume and quality** on performance and energy consumption.
- Data-centric properties are analyzed using **logarithmic curves**.

*(Insert Image: Example Logarithmic Curve)*

### Data Analysis and Modelling
- Compares **horizontal vs. vertical approaches** in data reduction.
- **Vertical approach** (reducing data per subset of nodes) is preferable for **lower energy consumption** and **higher accuracy**.
- A **Gradient Boosting Regressor** predicts necessary **data volume reduction** to maintain accuracy.

*(Insert Image: Accuracy vs. Data Volume Trade-off)*

*(Insert Image: Energy vs. Data Volume Trade-off)*

### FL Configuration Recommendation
- Recommends **optimal FL configurations** by selecting nodes and adjusting data volume.
- Uses a ranking system based on **carbon emissions and data quality**.
- Three selection methods compared:
  1. **Node Selection (NS)**: Selects nodes with highest ranking.
  2. **Minimal Smart Reduction (MSR)**: Removes "dirty" data while maintaining minimal nodes.
  3. **Smart Reduction (SR)**: Ensures sufficient clean data while minimizing emissions.

## Evaluation
### Experimental Setup
- Three **FL configurations** tested with different datasets.
- Compared **Baseline** (full FL training) vs. **NS, MSR, SR** methods.
- Each scenario was simulated **96 times**.

### Results
1. **Accuracy**: All methods maintained performance, with **SR achieving a 12% improvement**.
2. **Carbon Reduction**:
   - **NS: 56% average reduction (up to 90%)**.
   - **MSR: 45% average reduction**.
   - **SR: 25% average reduction**.

*(Insert Image: Accuracy Evaluation Results)*

*(Insert Image: Carbon Emissions Evaluation Results)*

## Conclusion
This thesis presents a **data-centric FL methodology** to optimize training while reducing environmental impact. The results show that **carbon emissions can be reduced by up to 90%** while maintaining or improving model performance. Future research will explore:
- **Automatic weight optimization** for node selection.
- **Additional data quality dimensions** for enhanced reduction strategies.

---

## Author
**Mattia Sabella**
## Advisor
**Prof. Monica Vitali**

---

## References
1. Martín Anselmo and Monica Vitali. *A data-centric approach for reducing carbon emissions in deep learning*, 2023.
2. Daniel J. Beutel et al. *Flower: A friendly federated learning research framework*, 2020.
3. Lukas Budach et al. *The effects of data quality on machine learning performance*, 2022.
4. Michaela Iorga et al. *The NIST definition of Fog Computing*, 2017.
[^5] 5. Roy Schwartz et al. *Green AI*, 2020.
6. Chen Zhang et al. *A survey on federated learning*, 2021.
