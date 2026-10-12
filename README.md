# Neural Caption-Image Retrieval

UE24CS352A Machine Learning, Mini-Project, Problem 36 (Image-caption retrieval)

**Team**
- Tatam Mahalakshmi (PES2UG24CS558)
- T S Sathya Kiran (PES2UG24CS547)

## What this project does

Given a text query such as "a man riding a bike next to a train", the system returns the most relevant images from a collection. Captions and images are mapped into the same 1024-dimensional space, and retrieval is a nearest-neighbour search in that space.

- **Images:** features from a pretrained VGG-19 (fc7 layer, 4096 numbers), reduced to 1024 by a linear layer
- **Text:** pretrained GloVe word vectors fed to a GRU or LSTM; the final state is the caption embedding
- **Training:** ranking loss with in-batch negatives (a caption must be closer to its own image than to other images in the batch, and the reverse)
- **Baseline:** average GloVe vector of the caption, mapped to VGG features with ridge regression

Based on "Neural Caption-Image Retrieval" by Qian and Lamberti (Stanford CS229).

## Dataset

MS-COCO captions, 5 captions per image. Everything is downloaded automatically by the notebook.

| Split | Images | Source |
|---|---|---|
| Train | 40,000 | COCO train2017 (random sample) |
| Validation | 500 | COCO val2017 |
| Test | 1,000 | COCO val2017 |

The test set has 5,000 queries (5 captions for each of its 1,000 images). No image is in more than one split.

## Repository contents

| File | Description |
|---|---|
| `caption_image_retrieval.ipynb` | Full pipeline: data, feature extraction, baseline, neural models, evaluation, export |
| `demo.ipynb` | Loads the trained model and searches the test images with your own text queries (no training) |
| `results/` | `results.csv`, training curves, caption-length plot, example retrievals |
| `docs/` | Write-up PDF and presentation |
| `requirements.txt` | Python packages |

## How to run

### 1. Train (Google Colab)

1. Open `caption_image_retrieval.ipynb` in Google Colab.
2. Set the runtime to GPU (Runtime > Change runtime type > T4).
3. Run all cells from the top and allow access to Google Drive when asked. Drive is used to cache extracted features and finished models, so after a disconnect you can rerun from the top and it resumes.

Roughly 1 to 1.5 hours on a T4. The notebook downloads the COCO annotations, streams the images, downloads GloVe, extracts features, trains four neural models, evaluates all models on the test set and writes results to `outputs/`. The last sections create `demo_bundle.zip` (trained model and test gallery) and copy it to `MyDrive/demo_bundle.zip`.

### 2. Demo without retraining

1. Download the trained bundle: **[demo_bundle.zip](https://drive.google.com/file/d/11KbUwM87LbZNC1wxoW7pK6bzkVzY2N1q/view?usp=sharing)**
2. Open `demo.ipynb`, either in Colab or locally.
3. Colab: upload the zip to your Drive as `MyDrive/demo_bundle.zip`, then run all cells. Local: put `demo.ipynb` and `demo_bundle.zip` in the same folder and run all cells (CPU is enough).
4. Edit the `query = "..."` cell and run it to search. The baseline's results are shown above the neural model's.

### Local setup for the demo

```
pip install -r requirements.txt
jupyter notebook demo.ipynb
```

## Results

Test set: 1,000 images, 5,000 queries. R@K is the percentage of queries whose correct image is among the top K results. Rank columns are the median and mean rank of the correct image out of 1,000.

| Method | R@1 | R@5 | R@10 | Median rank | Mean rank |
|---|---|---|---|---|---|
| Baseline (ridge regression on mean GloVe) | 4.4 | 13.7 | 20.6 | 82.0 | 190.1 |
| GRU (random word vectors) | 30.3 | 64.6 | 79.5 | 3.0 | 11.4 |
| GRU + GloVe | 29.5 | 65.1 | 80.4 | 3.0 | 10.2 |
| LSTM + GloVe | 31.1 | 66.2 | 80.3 | 3.0 | 10.2 |
| **GRU + GloVe (order embedding)** | **32.3** | **67.1** | **80.7** | **3.0** | **9.7** |

The best model was chosen on the validation set. The neural models are far better than the baseline, and the baseline gets worse as captions get longer while the neural models stay flat (see `results/length.png`). The four neural variants are within about 1 point of each other and were each trained once, so the differences between them are small.

## Settings

Adam, learning rate 2e-4, batch size 128, dropout 0.2, gradient clipping 2.0, at most 25 epochs with early stopping (patience 5) on validation R@10. Embedding size 1024, GloVe 300-d, margin 0.2 (Euclidean) or 0.05 (order embedding). Single center crop for VGG features.

## Limitations

- Trained on 40,000 COCO images, not the full dataset
- VGG features are frozen
- One training run per model
- Rare words and spatial relations ("in front of") are handled poorly
- The demo searches only the 1,000 test images
