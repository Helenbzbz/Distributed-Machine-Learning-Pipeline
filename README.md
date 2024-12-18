# **Distributed Machine Learning Pipeline**

**Author**: Helen Zheng  
**Net-ID**: jz855  
**Date**: 2024-12-18  

This project implements a **distributed machine learning pipeline** using **gRPC-based communication** for training a **Multi-Layer Perceptron (MLP)** model on the **MNIST dataset**. It simulates GPU device servers for distributed computation and a coordinator server for orchestration. The **AllReduce Ring Algorithm** is used for gradient synchronization during training.

---

## **Table of Contents**

1. [Setup](#setup)  
2. [Components](#components)  
   - [GPU Coordinator Server](#gpu-coordinator-server)  
   - [GPU Device Server](#gpu-device-server)  
3. [ML Training and Testing](#ml-training-and-testing)  
4. [Pipeline Execution](#pipeline-execution)  

---

## **1. Setup**

### Prerequisites

1. **Install Go** (v1.18 or higher):  
   [Download Go](https://golang.org/doc/install)

2. **Install Protocol Buffers**:  
   ```bash
   go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
   go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
   ```

3. **Install Dependencies**:  
   Navigate to the project directory and run:
   ```bash
   cd DSML
   go mod tidy
   ```

4. **Download MNIST Dataset**:  
   Ensure the following files are located in the `data/` directory (source: [MNIST Datasets](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/)):
   - `train-images-idx3-ubyte.gz`  
   - `train-labels-idx1-ubyte.gz`  
   - `t10k-images-idx3-ubyte.gz`  
   - `t10k-labels-idx1-ubyte.gz`

---

## **2. Components**

### **2.1. GPU Coordinator Server**

The **GPU Coordinator Server** is responsible for:
- **CommInit**: Initializing communication with device servers.
- **AllReduceRing**: Synchronizing gradients using the AllReduce Ring algorithm.
- **Health Monitoring**: Checking the availability of connected devices.

#### **Tests for GPU Coordinator Server**

The following tests validate the GPU Coordinator Server:

1. **TestAllReduceComparison**: Compares the performance of Naive AllReduce and AllReduce Ring implementations. It highlights the latency improvement (Naive: 83ms, Ring: 8ms).
2. **TestCommInitWithInvalidDevices**: Verifies that invalid devices are correctly rejected during communication initialization.
3. **TestMemcpyHostToDeviceAndDeviceToHost**: Tests memory copy operations between host and device.
4. **TestGroupOperationsWithoutComm**: Checks behavior of group operations without active communication.
5. **TestCommDestroyInvalidId**: Ensures safe destruction of communicators with invalid IDs.
6. **TestAllReduceRingSingleDevice**: Validates AllReduce Ring functionality with a single device.
7. **TestCoordinatorDeviceFailure**: Simulates device failures and ensures proper error handling during AllReduce.

Run the test suite to validate the Coordinator Server:
```bash
cd DSML/gpu_coordinator_service
go test -v
```
**Expected Output:**
```bash
=== RUN   TestAllReduceComparison
    allreduce_comparison_test.go:75: CommInit successful: CommId=0, Devices=3
2024/12/18 10:35:36 Naive AllReduce completed: DataSize=1048576, Latency=10ms, TotalTime=83ms, TotalDataTransferred=6291456 bytes
2024/12/18 10:35:36 AllReduceRing step completed successfully for Communicator 0
--- PASS: TestAllReduceComparison (0.17s)
... (additional test outputs) ...
PASS
ok      example.com/dsml/gpu_coordinator_service        6.706s
```
This output shows a latency comparison between the two algorithms:

- **Naive AllReduce**: Latency of 83ms.
- **AllReduce Ring**: Latency of 8ms.

The key difference in their implementations lies in how data synchronization is handled:
1. **Naive AllReduce**: Each device communicates with every other device, leading to higher communication overhead.
2. **AllReduce Ring**: Devices are arranged in a logical ring, reducing communication steps to a minimum. This significantly lowers latency and improves efficiency.

---

### **2.2. GPU Device Server**

The **GPU Device Server** simulates individual GPU devices. Each server handles:
- **Memory Operations**: Transferring data between the host and device using `Memcpy`.
- **Stream Communication**: Managing send and receive data streams.
- **Metadata Retrieval**: Providing device memory information.

#### **Tests for GPU Device Server**

The following tests validate the GPU Device Server:

1. **TestGetDeviceMetadata**: Ensures correct retrieval of device metadata.
2. **TestBeginSend**: Tests initiating data sends from the device.
3. **TestBeginReceive**: Verifies data receive operations on the device.
4. **TestStreamSend**: Validates data stream send operations.
5. **TestGetStreamStatus**: Ensures proper reporting of stream statuses.

Run tests for the GPU Device Server:
```bash
cd DSML/gpu_device_service
go test -v
```
**Expected Output:**
```bash
=== RUN   TestGetDeviceMetadata
--- PASS: TestGetDeviceMetadata (0.00s)
... (additional test outputs) ...
PASS
ok      example.com/dsml/gpu_device_service     0.493s
```

---

## **3. ML Training and Testing**

The **client** manages the training and evaluation of the MLP model on the MNIST dataset. The MLP architecture includes:

- **Input Layer**: 784 input nodes, corresponding to the flattened 28x28 pixel MNIST images.
- **Hidden Layers**: Two hidden layers:
  - **Layer 1**: 128 neurons with ReLU activation.
  - **Layer 2**: 64 neurons with ReLU activation.
- **Output Layer**: 10 output nodes, corresponding to the 10 possible digits (0-9), with softmax activation.

Training parameters include:
- **Batch Size**: 64 samples per batch.
- **Learning Rate**: 0.01, with an adaptive learning rate scheduler.
- **Optimizer**: Stochastic Gradient Descent (SGD).
- **Epochs**: 10 iterations over the dataset.

The process performs the following steps:

1. **Load MNIST Dataset**: Preprocess training and testing data.
2. **Forward and Backward Pass**: Execute MLP computations for each batch.
3. **Gradient Synchronization**: Aggregate gradients using the Coordinator Server.
4. **Model Updates**: Apply synchronized gradients to update weights.
5. **Testing**: Evaluate model accuracy on the test dataset.

---

## **4. Pipeline Execution**

### **Step 1: Start GPU Device Servers**

Launch three GPU Device Servers on ports **5003**, **5004**, and **5005**:
```bash
cd DSML/cmd/gpu_device_server
go run main.go
```
Each server runs on a unique port with its own **Device ID**.
**Expected Output:**
```bash
2024/12/18 10:40:12 GPU Device server listening on port 5003 with device ID 1
2024/12/18 10:40:12 GPU Device server listening on port 5004 with device ID 2
2024/12/18 10:40:12 GPU Device server listening on port 5005 with device ID 3
```

### **Step 2: Start the GPU Coordinator Server**

In a separate terminal, start the Coordinator Server:
```bash
cd DSML/cmd/gpu_coordinator_server
go run main.go
```
The server listens on **port 50051**.
**Expected Output:**
```bash
2024/12/18 10:40:30 GPU Coordinator server listening on port 50051
```

### **Step 3: Run the Client for Training**

Open a new terminal and execute the client script:
```bash
cd DSML
go run client/client.go
```
**Expected Output:**
```bash
Loaded 60000 training samples
Loaded 10000 test samples
2024/12/18 10:41:30 Starting MLP training...
2024/12/18 10:42:50 Epoch 1 complete: Avg Loss: 1.3723, Accuracy: 71.94%
2024/12/18 10:44:01 Epoch 2 complete: Avg Loss: 0.5282, Accuracy: 86.72%
... (output for subsequent epochs) ...
2024/12/18 10:53:42 Training complete.
Final Test Accuracy: 92.89%
```

This indicates the MLP model achieves a test accuracy of **92.89%** after 10 epochs.

