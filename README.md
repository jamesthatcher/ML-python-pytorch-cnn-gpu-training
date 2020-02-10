# ML-python-sklearn-svm-cpu-training


Quickstart project for executing a Iris classifier using the SciKit-Learn framework on a CPU.

This Quickstart trains the model and persists as in ONNX format. The service runtime will then serve the model on localhost where the user can then send GET requests to perform inference.


# Tensorboard metrics
Tensorboard logs can be viewed by running `tensorboard --logdir=/metrics/ --port 8080`

### Scalar recordings

Average loss and accuracy metrics are captured at the end of each training epoch for the train/val datasets. These can be inspected in the Scalar graph in Tensorboard.


### Histogram recordings

### Images recordings

###

###