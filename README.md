# Multi-Distillation from Speech and Music Representation Models

This repository contains the official implementation for our ASRU 2025 submission:  
**Multi-Distillation from Speech and Music Representation Models**.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Datasets & Teacher Models](#datasets--teacher-models)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Overview

Real-world audio often mixes speech and music, but most models process only one domain.  
We propose a multi-teacher distillation framework to unify speech and music models into a single compact model, leveraging HuBERT/WavLM (for speech) and MERT (for music) as teachers.  
Our model performs well on both domains and shows improved few-shot generalization.

---

## Features

- Multi-teacher distillation (speech + music SSL models)
- Data-domain separation and hybrid feature translators
- Adaptive loss weighting for balanced learning
- Comprehensive evaluation on speech and music benchmarks (SUPERB, MARBLE)
- Few-shot learning experiments

---

## Getting Started

1. **Install Dependencies**  
   (Provide `requirements.txt` or list main packages)

2. **Download Teacher Model Checkpoints**
   - [HuBERT-base (ls960)](https://huggingface.co/facebook/hubert-base-ls960)
   - [WavLM-base+](https://huggingface.co/microsoft/wavlm-base-plus)
   - [MERT-public-v0](https://huggingface.co/m-a-p/MERT-v0-public)

3. **Prepare Data**
   - [LibriSpeech](https://www.openslr.org/12)
   - [Music4All](https://github.com/MTG/Music4All)

4. **Run Training**
   ```bash
   python train.py --config configs/multi_distill.yaml
