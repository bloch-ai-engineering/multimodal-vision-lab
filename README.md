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

**Result: 17% accuracy.**
- **What the notebook concluded:** CLIP is a general-purpose model, not a face model; angle and lighting vary; and there are few reference images.
- **What a later review found:** the faces are passed to CLIP as flattened pixel vectors (`lfw_people.data`) rather than 2-D images (`lfw_people.images`). The transformers "ambiguous channel dimension" warnings in the output point to this. Feeding 2-D images is the first fix to try.

## Takeaways

- Captioning is solid on clean, typical scenes and weakest on cluttered scenes, small objects and colours.
- VQA does well on salient-object questions and poorly on counting and fine-grained attributes.
- For identity tasks, use embeddings trained for faces (e.g. ArcFace) and check that the preprocessing matches what the model expects. A general-purpose CLIP is a weak baseline.

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
