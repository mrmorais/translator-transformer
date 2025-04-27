# Transformer based PT-EN Translator

A straightforward implementation of a Transformer model for Translation of Portuguese to English sentences.

- Model Card
- [pretrained weights (.zip)](https://drive.google.com/file/d/121j-x9wKa2kIVl-xCUrOIJveCLTjuR5O/view)

Install ray on Apple Silicon: https://docs.ray.io/en/latest/ray-overview/installation.html#m1-mac-apple-silicon-support

This README only contain how-tos and project setup instructions. Later I intent to write a blog post on the transformer arch and learnings from this project.

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
