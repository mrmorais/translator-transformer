# Transformer based translator model

A straightforward implementation of a Transformer model for the translation task, trained on Portuguese to English translation.

- [Model Card](./model_card.md)
- [pretrained weights (.zip)](https://drive.google.com/file/d/121j-x9wKa2kIVl-xCUrOIJveCLTjuR5O/view)

Install ray on Apple Silicon: https://docs.ray.io/en/latest/ray-overview/installation.html#m1-mac-apple-silicon-support

This README only contain how-tos and project setup instructions. Later I intent to write a blog post on the transformer arch and learnings from this project.

## Model

The ~250k tuples are tokenized in a BERT-style [tokenizer](./translator/tokenizer.py) with reduced vocab sizes of 28576 (eng.) and 30522 (por.) The ensemble encoder-decoder Transformer uses default parameters: 8 self-attention heads, 6 enc/dec layers and 512-d embedding space.

## Scaling training

Ray is a framework that enables distributed training of the model on a Ray cluster. I used the utility CLI command to deploy the auto-scalable cluster on AWS. The config is at `aws-cluster.yml`.

```sh
ray up aws-cluster.yml
```

It takes a few minutes for the head to be available. Ray will spin a EC2 head node. When a job is dispatched this node will do the provisioning of workers nodes and setup the environment dependencies. For my local setup I simply attached the head node with:

```sh
ray attach -p 8265 aws-cluster.yml
```

The port forwarding allows me to access the cluster dashboard (http://localhost:8265) and use this address to submit a new job. The submission step could be done with CLI, but its more convenient to do so via Python script (`submit_train.py`)

Run `ray down aws-cluster.yml` to shutdown the cluster. Some residual resources may still be up, make sure to remove them.

## Train and infer

Use `train_model.py` and the `infer_from_artifact.ipynb` for non-distributed training and inference testing.

## Samples and observations

With very few training what this simple model was able to learn?

Surprisingly, it captures the grammatical structure quite well. Some instances:

> eles conseguiram ir ao evento? -> did they get to the event?

> eu não iria à festa se eu fosse ele -> i wouldn't go to the party if i were him.

> a casa do rei é um... -> the king's house is a...

### Alignment

Languages may have different rules on how words are ordered, nouns and adjectives are example. Neural translation needs spatial context in order to learn those rules (like RNN model), attention models are better at learning them given the long-range retrieval and multi head approach:

> o pequeno gato cinza -> the little gray cat.

In some cases it deviates from the original meaning:

> a grande casa amarela -> the big house is yellow. ❌

### Nonsenses

Generally creating wrong translations.

> você que inventou de inventar ->  you've made fun of protein.

> papel e caneta na mão -> paper is a pen in the hand.

> A Europa é o principal destino turístico do mundo -> europe is the main tourist tourist tourist of the world.

This happens more often with complex or long sentences. Certainly the small dataset (250k) constraints the model vocabulary and generalization; and the decision to train on small sentences only. More training steps surely could improve some deficits, but the overall objective of the model were achieved (validating the implementation per se)
