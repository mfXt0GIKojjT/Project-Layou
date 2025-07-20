# Temporal Modeling of Patient Education Engagement Using Health Information Access

This repository contains the implementation of **Temporal Latent Gradient Network (TLGN)**, a framework designed to model patient education engagement dynamics using health information access logs. The goal is to enhance predictive accuracy, interpretability, and robustness in healthcare interventions.

## 🧠 Overview

With the rapid digital transformation of healthcare, effectively engaging patients through educational content is crucial for better outcomes. However, traditional modeling approaches often struggle to handle:

- Long-range dependencies in user behavior
- Irregular and sparse data
- Noise and biases in access patterns
- Causal influences from domain-specific signals

**TLGN** addresses these limitations by:

- Learning latent temporal trajectories from gradients in a non-autoregressive way
- Modeling long-term engagement patterns using a neural latent dynamics model
- Integrating the **Causal-Aware Temporal Denoising (CATD)** strategy

## 🛠 Features

- **Causal-aware denoising** using domain knowledge and structural constraints
- **Robust forecasting** with temporal alignment
- **Compatibility with both synthetic and real-world datasets**
- Outperforms traditional temporal models in both **accuracy** and **interpretability**

## 📊 Datasets

The model has been validated on:

- **Synthetic datasets** simulating health information access behavior
- **Real-world patient education engagement datasets**

## 📦 Repository Structure

