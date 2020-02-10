# ML-python-pytorch-cnn-gpu-training

Quickstart project for executing an MNIST classifier using the PyTorch framework on a GPU.

This quickstart trains the model and persists as in ONNX format. The service runtime will then serve the model on localhost where the user can then send GET requests to perform inference.


# Tensorboard metrics
Metrics for this quickstart are recorded in the Tensorboard logs.

Tensorboard logs can be viewed by running `tensorboard --logdir=/metrics/ --port 8080`

### Scalar recordings

Average loss and accuracy metrics are captured at the end of each training epoch for the train/val datasets. These can be inspected in the Scalar graph.