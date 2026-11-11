# Multimodal Vision Lab

**Vision-language models in practice: image captioning with BLIP, visual question answering with BLIP-VQA, and a small face-recognition system built on CLIP embeddings. Each comes with a failure analysis of where the models break.**

![Python](https://img.shields.io/badge/python-3.11+-blue) ![Transformers](https://img.shields.io/badge/🤗%20Transformers-BLIP%20%C2%B7%20CLIP-yellow) ![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c)

## 1. Image captioning: `Salesforce/blip-image-captioning-base`

Captions for 5 images covering people, animals, an indoor scene, an outdoor scene, and a cluttered table (in [`images/`](images/)).

| Failure | What went wrong |
|---|---|
| *"a room with a yellow couch and a blue rug"* | The couch is pink/beige. The model took the colour from a yellow hutch nearby and left out most of the objects. |
| *"a group of birds standing on top of a dirt field"* | The ground is grass with patches of dirt. The model picked the wrong one. |

## 2. Visual question answering: `Salesforce/blip-vqa-capfilt-large`

3 questions per image (object recognition, counting, colour or attribute) across 5 images.

| Question | Model | Truth | Why |
|---|---|---|---|
| How many birds are there? | 6 | 4 | Birds overlap each other. Counting is a classic VQA weak spot |
| What is on the table? | toys | stationery and electronics | A cluttered scene, read at the wrong level of abstraction |

## 3. Face recognition with CLIP embeddings

A few-shot identity matcher on [LFW](https://vis-www.cs.umass.edu/lfw/):
- 3 identities, split into reference and query images.
- Each image is embedded with `openai/clip-vit-base-patch32`.
- A query gets the identity with the highest average cosine similarity, or "Unknown" if that similarity falls below a threshold.

**Result: 100% accuracy (6 of 6 query faces).**

The first version scored **17%**. The cause was a preprocessing bug, not CLIP itself: faces were passed to the model as flattened 2,914-pixel vectors (`lfw_people.data`) instead of 2-D images (`lfw_people.images`). That garbled every embedding, and the transformers "ambiguous channel dimension" warnings in the output were the clue. After switching to the 2-D grayscale images and converting them to RGB properly, every query face matches its identity.

The test set is small (2 queries per identity), so this shows the pipeline works. It isn't a benchmark.

## Takeaways

- Captioning is solid on clean, typical scenes and weakest on cluttered scenes, small objects and colours.
- VQA does well on salient-object questions and poorly on counting and fine-grained attributes.
- For identity tasks, **check the preprocessing first**. One wrong array shape cost 83 points of accuracy. With correct inputs, general-purpose CLIP embeddings are a strong few-shot baseline. Face-specific embeddings (e.g. ArcFace) are the next step for harder, larger galleries.

## Run it

```bash
pip install -r requirements.txt
cd notebooks && jupyter notebook captioning_vqa_face_recognition.ipynb
```

The notebook reads images from `../images/`, and the LFW dataset downloads automatically through scikit-learn. It was developed on a Colab T4.

## Context

This was homework 4 of the **LLM Architectures** module in **Nebius Academy: AI Performance Engineering**. The task formulations are kept inline in the notebook.

## Authors

**Nathanael Bloch** · **Tomer Porat**
