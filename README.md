# ClientServer-ML-RDD

This repository demonstrates how to simulate the process of sending and receiving data using the CIFAR-10 dataset. A machine learning model will be used for training with Spark RDD.

The Spark Streaming process involves the following stages: First, the sender (stream.py) will be used to send the data. Then, the receiver (main.py) will receive and process the data. Online learning will be performed using SGDClassifier (SGDC) to continuously predict performance on each batch of data.

## Installation

Clone the repository:
   ```bash
   git clone https://github.com/HuynhToan2004/ClientServer-ML-RDD.git
   cd ClientServer-ML-RDD
   ```

## Directory Structure

```plaintext
ClientServer-ML-RDD/
├── src/                              # Source code for the simulation
│   ├── models/                       # Machine Learning models and training scripts
│   │   ├── pac.py                    # Script implementing Stochastic Gradient Descent for online learning
│   ├── transforms_scr/               # Data transformation scripts
│   │   ├── apply_transform.py        # Script for applying transformations to data before streaming (Currently not used, but will be updated in the future)
│   ├── stream.py                     # Sender for streaming data
│   ├── main.py                       # Receiver for processing and training
└── data/                             # CIFAR-10 dataset and other related data

```

## How to run

1. Run the sender:
To start sending data, use the following command to execute stream.py (sender):
  ```bash
  spark-submit "src/stream.py" --folder /path/to/data --batch-size 32
  ```
• Adjust the --folder parameter to point to the directory containing your CIFAR-10 dataset.

• Modify the --batch-size parameter as desired.

2. Run the receiver:
To start receiving and processing data, use the following command to execute main.py (receiver):
  ```bash
  spark-submit "src/main.py"
  ```
This will run the receiver, which will continuously process the incoming data and apply online learning with the SGDClassifier.
## Note

• Ensure that Spark is installed and configured properly before running the scripts.

• You may need to adjust the Spark settings for your environment, such as the number of executors or memory allocation, depending on your system configuration.

