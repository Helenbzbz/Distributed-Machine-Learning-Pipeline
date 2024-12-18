# **Distributed Machine Learning Pipeline**

**Author**: Helen Zheng  
**Net-ID**: jz855  
**Date**: 2024-12-18  

This project implements a **distributed machine learning pipeline** using **gRPC-based communication** for training a **Multi-Layer Perceptron (MLP)** model on the **MNIST dataset**. It simulates GPU device servers for distributed computation and a coordinator server for orchestration. The **AllReduce Ring Algorithm** synchronizes gradients during training.

---

## **Table of Contents**

1. [Setup](#setup)  
2. [Components](#components)  
   - [GPU Coordinator Server](#gpu-coordinator-server)  
   - [GPU Device Server](#gpu-device-server)  
3. [ML Training and Testing](#ml-training-and-testing)  
4. [Testing the Components](#testing-the-components)  

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

3. **Dependencies**:  
   Use the `go.mod` file to install project dependencies:  
   ```bash
   cd DSML
   go mod tidy
   ```

4. **MNIST Dataset**:  
Ensure these files are in the `data/` directory (source: `https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/`):
- `train-images-idx3-ubyte.gz`  
- `train-labels-idx1-ubyte.gz`  
- `t10k-images-idx3-ubyte.gz`  
- `t10k-labels-idx1-ubyte.gz`  

---

## **2. Components**

### **2.1. GPU Coordinator Server**

The **GPU Coordinator Server** orchestrates distributed training by:  
- **CommInit**: Initializing communication with device servers.  
- **AllReduceRing**: Synchronizing gradients using the AllReduce Ring algorithm.  
- **Health Monitoring**: Checking the availability of connected devices.


#### **Test the GPU Coordinator Server**  

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
2024/12/18 10:35:36 AllReduceRing step completed successfully for Communicator 0
2024/12/18 10:35:36 AllReduceRing step completed successfully for Communicator 0
2024/12/18 10:35:36 AllReduceRing step completed successfully for Communicator 0
    allreduce_comparison_test.go:127: 
        --- AllReduce Comparison ---
    allreduce_comparison_test.go:128: Naive AllReduce: Time=126ms, DataTransferred=6291456 bytes
    allreduce_comparison_test.go:129: Ring AllReduce:  Time=8ms
2024/12/18 10:35:36 Communicator 0 destroyed
--- PASS: TestAllReduceComparison (0.17s)
=== RUN   TestCommInitWithInvalidDevices
--- PASS: TestCommInitWithInvalidDevices (0.01s)
=== RUN   TestMemcpyHostToDeviceAndDeviceToHost
2024/12/18 10:35:36 Communicator 0 destroyed
--- PASS: TestMemcpyHostToDeviceAndDeviceToHost (0.01s)
=== RUN   TestGroupOperationsWithoutComm
--- PASS: TestGroupOperationsWithoutComm (0.00s)
=== RUN   TestCommDestroyInvalidId
--- PASS: TestCommDestroyInvalidId (0.00s)
=== RUN   TestCoordinatorAllReduceRing
2024/12/18 10:35:36 Communicator 0 destroyed
--- PASS: TestCoordinatorAllReduceRing (0.00s)
=== RUN   TestAllReduceRingSingleDevice
2024/12/18 10:35:36 Communicator 0 destroyed
--- PASS: TestAllReduceRingSingleDevice (0.01s)
=== RUN   TestCoordinatorDeviceFailure
2024/12/18 10:35:41 Device 1 at 127.0.0.1:59515 unreachable: rpc error: code = Unavailable desc = connection error: desc = "transport: Error while dialing: dial tcp 127.0.0.1:59515: connect: connection refused"
2024/12/18 10:35:41 Communicator 0 lost devices; marking as FAILED
--- PASS: TestCoordinatorDeviceFailure (6.01s)
PASS
ok      example.com/dsml/gpu_coordinator_service        6.706s
```

We get an comparison between `NaiveAllReduce`, 126ms vs `RingAllReduce`, 8ms, which shows RingAllReduce reduces latency by 93.65% comparing to NaiveAllReduce.

---

### **2.2. GPU Device Server**

The **GPU Device Servers** simulate individual GPU devices. Each server handles:  
- **Memory Operations**: Using `Memcpy` to transfer data between the host and device.  
- **Stream Communication**: Managing data streams for send and receive operations.  
- **Metadata Retrieval**: Returning device memory information.

#### **Test the GPU Device Server**  
To run tests for the GPU Device Server:  
```bash
cd DSML/gpu_device_service
go test -v
```
**Expected Output:**
```base
=== RUN   TestGetDeviceMetadata
--- PASS: TestGetDeviceMetadata (0.00s)
=== RUN   TestBeginSend
--- PASS: TestBeginSend (0.00s)
=== RUN   TestBeginReceive
--- PASS: TestBeginReceive (0.00s)
=== RUN   TestStreamSend
--- PASS: TestStreamSend (0.00s)
=== RUN   TestGetStreamStatus
--- PASS: TestGetStreamStatus (0.00s)
PASS
ok      example.com/dsml/gpu_device_service     0.493s
```

---

## **3. ML Training and Testing**

The **client** runs the MLP model training and performs the following tasks:  
1. **Load MNIST Dataset**: Preprocesses training and testing data.  
2. **Forward and Backward Pass**: Performs MLP computations.  
3. **Gradient Synchronization**: Synchronizes gradients using the Coordinator Server.  
4. **Model Updates**: Updates weights after aggregated gradients.  
5. **Testing**: Evaluates the final model accuracy.

---

### **Steps to Run the Pipeline**

1. **Start GPU Device Servers**:  

Start 3 GPU Device Servers on ports **5003, 5004, and 5005**:  

```bash
cd DSML/cmd/gpu_device_server
go run main.go
```
Each device will run on a unique port with its own **Device ID**.
**Expected Output**
```bash
2024/12/18 10:40:12 GPU Device server listening on port 5003 with device ID 1
2024/12/18 10:40:12 GPU Device server listening on port 5004 with device ID 2
2024/12/18 10:40:12 GPU Device server listening on port 5005 with device ID 3
```

2. **Start the GPU Coordinator Server** (in separate terminal):  
   
```bash
cd DSML/cmd/gpu_coordinator_server
go run main.go
```
The server listens on **port 50051**.
**Expected Output**
```bash
2024/12/18 10:40:30 GPU Coordinator server listening on port 50051
```

3. **Run the Client for Training**:  
   Open a new terminal and execute:  
   ```bash
   cd DSML
   go run client/client.go
   ```

**Expected Output**
```bash

```



