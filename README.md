# Learning Facial Expression and Body Gesture Visual Information for Video Emotion Recognition

PyTorch implementation for the paper:

- Title: Learning Facial Expression and Body Gesture Visual Information for Video Emotion Recognition

- Authors: Jie Wei, Guanyu  Hu,  Xinyu Yang, Luu Anh Tuan, Yizhuo Dong

- Submitted to: EXPERT SYSTEMS WITH APPLICATIONS

- Abstract: Recent research has shown that facial expressions and body gestures are two significant implications in identifying human emotions. However, these studies mainly focus on contextual information of adjacent frames, and rarely explore the spatio-temporal relationships between distant or global frames. In this paper, we revisit the facial expression and body gesture emotion recognition problems, and propose to improve the performance of video emotion recognition by extracting the spatio-temporal features via further encoding temporal information. Specifically, for facial expression, we propose a super image-based spatio-temporal convolutional model (SISTCM) and a two-stream LSTM model to capture the local spatio-temporal features and learn global temporal cues of emotion changes. For body gestures, a novel representation method and an attention-based channel-wise convolutional model (ACCM) are introduced to learn key joints features and independent characteristics of each joint. Extensive experiments on five common datasets are carried out to prove the superiority of the proposed method, and the results proved learning two visual information leads to significant improvement over the existing sota methods.  

## Getting Started

```git
git clone https://github.com/Janie1996/FER_BGER.git
```

## Requirements

You can create an anaconda environment with:

```
conda env create -f environment.yaml
conda activate MSRFG
```

## Usage

### 1. Preparation
