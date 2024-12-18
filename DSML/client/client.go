package main

import (
	"compress/gzip"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	pb "example.com/dsml/proto"
	"github.com/schollz/progressbar/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Hyperparameters
const (
	batchSize       = 64
	inputSize       = 784 // MNIST: 28x28 images flattened
	hiddenSize      = 128
	outputSize      = 10 // 0-9 digits
	learningRate    = 0.01
	numEpochs       = 10
	gradientMemAddr = 0x1000
	weightMemAddr   = 0x2000
	dataMemAddr     = 0x3000
	labelMemAddr    = 0x4000
)

// MLP structure
type MLP struct {
	weights1 []float32
	bias1    []float32
	weights2 []float32
	bias2    []float32
}

func NewMLP() *MLP {
	rand.Seed(time.Now().UnixNano())
	initWeights := func(size int) []float32 {
		w := make([]float32, size)
		for i := range w {
			w[i] = (rand.Float32() - 0.5) * 0.1
		}
		return w
	}
	return &MLP{
		weights1: initWeights(inputSize * hiddenSize),
		bias1:    initWeights(hiddenSize),
		weights2: initWeights(hiddenSize * outputSize),
		bias2:    initWeights(outputSize),
	}
}

func floatsToBytes(floats []float32) []byte {
	buf := make([]byte, len(floats)*4)
	for i, f := range floats {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

func bytesToFloats(b []byte) []float32 {
	floats := make([]float32, len(b)/4)
	for i := range floats {
		floats[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return floats
}

func softmax(logits []float32) []float32 {
	maxLogit := float32(-1e9)
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}
	var sumExp float32
	for i, v := range logits {
		expV := float32(math.Exp(float64(v - maxLogit)))
		logits[i] = expV
		sumExp += expV
	}
	for i := range logits {
		logits[i] /= sumExp
	}
	return logits
}

func relu(x []float32) []float32 {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
	}
	return x
}

func reluBackward(dout, x []float32) {
	for i := range dout {
		if x[i] <= 0 {
			dout[i] = 0
		}
	}
}

func forwardPass(mlp *MLP, inputs []float32) (logits, hidden, z1 []float32) {
	z1 = make([]float32, batchSize*hiddenSize)
	for b := 0; b < batchSize; b++ {
		for h := 0; h < hiddenSize; h++ {
			sum := float32(0)
			for i := 0; i < inputSize; i++ {
				sum += inputs[b*inputSize+i] * mlp.weights1[i*hiddenSize+h]
			}
			sum += mlp.bias1[h]
			z1[b*hiddenSize+h] = sum
		}
	}

	hidden = make([]float32, len(z1))
	copy(hidden, z1)
	relu(hidden)

	logits = make([]float32, batchSize*outputSize)
	for b := 0; b < batchSize; b++ {
		for o := 0; o < outputSize; o++ {
			sum := float32(0)
			for h := 0; h < hiddenSize; h++ {
				sum += hidden[b*hiddenSize+h] * mlp.weights2[h*outputSize+o]
			}
			sum += mlp.bias2[o]
			logits[b*outputSize+o] = sum
		}
	}
	return logits, hidden, z1
}

func backwardPass(mlp *MLP, inputs, hidden, z1, logits, labels []float32) (dW1, dB1, dW2, dB2 []float32, loss float32) {
	for b := 0; b < batchSize; b++ {
		start := b * outputSize
		subLogits := logits[start : start+outputSize]
		softmax(subLogits)
		for c := 0; c < outputSize; c++ {
			if labels[b*outputSize+c] > 0 {
				l := -float32(math.Log(float64(subLogits[c]) + 1e-10))
				loss += l
			}
		}
	}
	loss /= float32(batchSize)

	dLogits := make([]float32, len(logits))
	copy(dLogits, logits)
	for i := range dLogits {
		dLogits[i] -= labels[i]
	}
	scale := float32(1.0 / float32(batchSize))
	for i := range dLogits {
		dLogits[i] *= scale
	}

	dW2 = make([]float32, hiddenSize*outputSize)
	dB2 = make([]float32, outputSize)
	for b := 0; b < batchSize; b++ {
		for o := 0; o < outputSize; o++ {
			dB2[o] += dLogits[b*outputSize+o]
			for h := 0; h < hiddenSize; h++ {
				dW2[h*outputSize+o] += hidden[b*hiddenSize+h] * dLogits[b*outputSize+o]
			}
		}
	}

	dHidden := make([]float32, len(hidden))
	for b := 0; b < batchSize; b++ {
		for h := 0; h < hiddenSize; h++ {
			sum := float32(0)
			for o := 0; o < outputSize; o++ {
				sum += dLogits[b*outputSize+o] * mlp.weights2[h*outputSize+o]
			}
			dHidden[b*hiddenSize+h] = sum
		}
	}

	reluBackward(dHidden, z1)

	dW1 = make([]float32, inputSize*hiddenSize)
	dB1 = make([]float32, hiddenSize)
	for b := 0; b < batchSize; b++ {
		for h := 0; h < hiddenSize; h++ {
			dB1[h] += dHidden[b*hiddenSize+h]
			for i := 0; i < inputSize; i++ {
				dW1[i*hiddenSize+h] += inputs[b*inputSize+i] * dHidden[b*hiddenSize+h]
			}
		}
	}
	return dW1, dB1, dW2, dB2, loss
}

func copyWeightsToDevice(ctx context.Context, deviceClient pb.GPUDeviceClient, deviceId uint64, mlp *MLP) {
	allWeights := append(mlp.weights1, append(mlp.bias1, append(mlp.weights2, mlp.bias2...)...)...)
	_, err := deviceClient.Memcpy(ctx, &pb.MemcpyRequest{
		Either: &pb.MemcpyRequest_HostToDevice{
			HostToDevice: &pb.MemcpyHostToDeviceRequest{
				HostSrcData: floatsToBytes(allWeights),
				DstDeviceId: &pb.DeviceId{Value: deviceId},
				DstMemAddr:  &pb.MemAddr{Value: weightMemAddr},
			},
		},
	})
	if err != nil {
		log.Fatalf("Failed to copy weights to device %d: %v", deviceId, err)
	}
}

func copyGradientsToDevice(ctx context.Context, deviceClients []pb.GPUDeviceClient, grads []float32) {
	for i, dev := range deviceClients {
		_, err := dev.Memcpy(ctx, &pb.MemcpyRequest{
			Either: &pb.MemcpyRequest_HostToDevice{
				HostToDevice: &pb.MemcpyHostToDeviceRequest{
					HostSrcData: floatsToBytes(grads),
					DstDeviceId: &pb.DeviceId{Value: uint64(i + 1)},
					DstMemAddr:  &pb.MemAddr{Value: gradientMemAddr},
				},
			},
		})
		if err != nil {
			log.Fatalf("Failed to send gradients to device %d: %v", i+1, err)
		}
	}
}

func retrieveGradientsFromDevice(ctx context.Context, deviceClients []pb.GPUDeviceClient, gradSize int) []float32 {
	dev := deviceClients[0]
	resp, err := dev.Memcpy(ctx, &pb.MemcpyRequest{
		Either: &pb.MemcpyRequest_DeviceToHost{
			DeviceToHost: &pb.MemcpyDeviceToHostRequest{
				SrcDeviceId: &pb.DeviceId{Value: 1},
				SrcMemAddr:  &pb.MemAddr{Value: gradientMemAddr},
				NumBytes:    uint64(gradSize * 4),
			},
		},
	})
	if err != nil {
		log.Fatalf("Failed to retrieve gradients from device: %v", err)
	}
	return bytesToFloats(resp.GetDeviceToHost().DstData)
}

func updateWeights(mlp *MLP, dW1, dB1, dW2, dB2 []float32) {
	for i := range mlp.weights1 {
		mlp.weights1[i] -= learningRate * dW1[i]
	}
	for i := range mlp.bias1 {
		mlp.bias1[i] -= learningRate * dB1[i]
	}
	for i := range mlp.weights2 {
		mlp.weights2[i] -= learningRate * dW2[i]
	}
	for i := range mlp.bias2 {
		mlp.bias2[i] -= learningRate * dB2[i]
	}
}

// Functions to load MNIST data
func LoadMNISTImages(path string) ([]float32, int) {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("Failed to open images file: %v", err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		log.Fatalf("Failed to create gzip reader: %v", err)
	}
	defer gz.Close()

	var magic, numImages, rows, cols int32
	if err := binary.Read(gz, binary.BigEndian, &magic); err != nil {
		log.Fatalf("Failed to read magic number: %v", err)
	}
	if magic != 2051 {
		log.Fatalf("Invalid magic number for images: %v", magic)
	}
	if err := binary.Read(gz, binary.BigEndian, &numImages); err != nil {
		log.Fatalf("Failed to read number of images: %v", err)
	}
	if err := binary.Read(gz, binary.BigEndian, &rows); err != nil {
		log.Fatalf("Failed to read rows: %v", err)
	}
	if err := binary.Read(gz, binary.BigEndian, &cols); err != nil {
		log.Fatalf("Failed to read cols: %v", err)
	}

	imageSize := int(rows * cols)
	images := make([]byte, imageSize*int(numImages))

	if _, err := io.ReadFull(gz, images); err != nil {
		log.Fatalf("Failed to read image data: %v", err)
	}

	floatImages := make([]float32, len(images))
	for i, v := range images {
		floatImages[i] = float32(v) / 255.0
	}

	return floatImages, int(numImages)
}

func LoadMNISTLabels(path string) ([]float32, int) {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("Failed to open labels file: %v", err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		log.Fatalf("Failed to create gzip reader: %v", err)
	}
	defer gz.Close()

	var magic, numLabels int32
	if err := binary.Read(gz, binary.BigEndian, &magic); err != nil {
		log.Fatalf("Failed to read label magic number: %v", err)
	}
	if magic != 2049 {
		log.Fatalf("Invalid magic number for labels: %v", magic)
	}
	if err := binary.Read(gz, binary.BigEndian, &numLabels); err != nil {
		log.Fatalf("Failed to read number of labels: %v", err)
	}

	labels := make([]byte, numLabels)
	if _, err := io.ReadFull(gz, labels); err != nil {
		log.Fatalf("Failed to read label data: %v", err)
	}

	floatLabels := make([]float32, int(numLabels)*outputSize)
	for i, label := range labels {
		floatLabels[i*outputSize+int(label)] = 1.0
	}

	return floatLabels, int(numLabels)
}

func computeAccuracy(logits, labels []float32) int {
	correct := 0
	for b := 0; b < batchSize; b++ {
		start := b * outputSize
		subLogits := logits[start : start+outputSize]

		// Find argmax predicted
		predClass := 0
		maxVal := float32(-1e9)
		for c := 0; c < outputSize; c++ {
			if subLogits[c] > maxVal {
				maxVal = subLogits[c]
				predClass = c
			}
		}

		// Find true label
		trueClass := -1
		for c := 0; c < outputSize; c++ {
			if labels[start+c] == 1.0 {
				trueClass = c
				break
			}
		}

		if trueClass == predClass {
			correct++
		}
	}
	return correct
}

func trainEpoch(ctx context.Context, client pb.GPUCoordinatorClient, deviceClients []pb.GPUDeviceClient, mlp *MLP, trainData, trainLabels []float32, commId uint64) {

	// Declare totalBatches before using it in progressbar
	totalBatchesInt := len(trainData) / (batchSize * inputSize)

	// Create a progress bar for batches within this epoch
	bar := progressbar.NewOptions(totalBatchesInt,
		progressbar.OptionSetDescription("Training Batches"),
		progressbar.OptionShowCount(),
		progressbar.OptionThrottle(65*time.Millisecond),
		progressbar.OptionClearOnFinish(),
		progressbar.OptionShowIts(),
	)

	var totalLoss float32
	var totalCorrect int
	var totalSamples int = totalBatchesInt * batchSize

	for batch := 0; batch < totalBatchesInt; batch++ {
		start := batch * batchSize * inputSize
		end := start + batchSize*inputSize
		batchData := trainData[start:end]

		labelStart := batch * batchSize * outputSize
		labelEnd := labelStart + batchSize*outputSize
		batchLabels := trainLabels[labelStart:labelEnd]

		logits, hidden, z1 := forwardPass(mlp, batchData)
		dW1, dB1, dW2, dB2, loss := backwardPass(mlp, batchData, hidden, z1, logits, batchLabels)
		totalLoss += loss

		// Compute accuracy for this batch
		batchLogits := make([]float32, len(logits))
		copy(batchLogits, logits)
		for b := 0; b < batchSize; b++ {
			sub := batchLogits[b*outputSize : b*outputSize+outputSize]
			softmax(sub)
		}
		correct := computeAccuracy(batchLogits, batchLabels)
		totalCorrect += correct

		allGrads := append(dW1, append(dB1, append(dW2, dB2...)...)...)
		copyGradientsToDevice(ctx, deviceClients, allGrads)

		_, err := client.AllReduceRing(ctx, &pb.AllReduceRingRequest{
			CommId: commId,
			Count:  uint64(len(allGrads) * 4),
		})
		if err != nil {
			log.Fatalf("AllReduceRing failed: %v", err)
		}

		aggGrads := retrieveGradientsFromDevice(ctx, deviceClients, len(allGrads))
		offset := 0
		dW1Agg := aggGrads[offset : offset+len(dW1)]
		offset += len(dW1)
		dB1Agg := aggGrads[offset : offset+len(dB1)]
		offset += len(dB1)
		dW2Agg := aggGrads[offset : offset+len(dW2)]
		offset += len(dW2)
		dB2Agg := aggGrads[offset : offset+len(dB2)]

		updateWeights(mlp, dW1Agg, dB1Agg, dW2Agg, dB2Agg)

		for i, dev := range deviceClients {
			copyWeightsToDevice(ctx, dev, uint64(i+1), mlp)
		}

		bar.Add(1)
	}

	bar.Finish()
	avgLoss := totalLoss / float32(totalBatchesInt)
	accuracy := float32(totalCorrect) / float32(totalSamples) * 100.0
	log.Printf("Epoch complete: Avg Loss: %.4f, Accuracy: %.2f%%", avgLoss, accuracy)
}

// Test the model after training using the test dataset with a progress bar
func testModel(mlp *MLP, testData, testLabels []float32) {
	// Declare totalTestBatches before using it in progressbar
	totalTestBatchesInt := len(testData) / (batchSize * inputSize)

	// Create a progress bar for testing batches
	bar := progressbar.NewOptions(totalTestBatchesInt,
		progressbar.OptionSetDescription("Testing Batches"),
		progressbar.OptionShowCount(),
		progressbar.OptionThrottle(65*time.Millisecond),
		progressbar.OptionClearOnFinish(),
		progressbar.OptionShowIts(),
	)

	var totalCorrectTest int
	var totalTestSamples int = totalTestBatchesInt * batchSize

	for batch := 0; batch < totalTestBatchesInt; batch++ {
		start := batch * batchSize * inputSize
		end := start + batchSize*inputSize
		batchData := testData[start:end]

		labelStart := batch * batchSize * outputSize
		labelEnd := labelStart + batchSize*outputSize
		batchLabels := testLabels[labelStart:labelEnd]

		logits, _, _ := forwardPass(mlp, batchData)
		for b := 0; b < batchSize; b++ {
			sub := logits[b*outputSize : b*outputSize+outputSize]
			softmax(sub)
		}
		correct := computeAccuracy(logits, batchLabels)
		totalCorrectTest += correct

		bar.Add(1)
	}

	bar.Finish()

	testAccuracy := float32(totalCorrectTest) / float32(totalTestSamples) * 100.0
	fmt.Printf("Final Test Accuracy: %.2f%%\n", testAccuracy)
}

func connectToDeviceServers(addresses []string) []pb.GPUDeviceClient {
	var deviceClients []pb.GPUDeviceClient
	for _, addr := range addresses {
		conn, err := grpc.Dial(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			log.Fatalf("Failed to connect to device server at %s: %v", addr, err)
		}
		deviceClients = append(deviceClients, pb.NewGPUDeviceClient(conn))
	}
	return deviceClients
}

func main() {
	// Step 1: Connect to GPU Coordinator
	coordinatorAddr := "localhost:50051"
	conn, err := grpc.Dial(coordinatorAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to coordinator: %v", err)
	}
	defer conn.Close()
	client := pb.NewGPUCoordinatorClient(conn)

	// Step 2: Connect to GPU Devices
	deviceAddresses := []string{"localhost:5003", "localhost:5004", "localhost:5005"}
	deviceClients := connectToDeviceServers(deviceAddresses)

	// Initialize Communicator
	ctx := context.Background()
	initResp, err := client.CommInit(ctx, &pb.CommInitRequest{
		NumDevices:      uint32(len(deviceAddresses)),
		DeviceAddresses: deviceAddresses,
	})
	if err != nil {
		log.Fatalf("CommInit failed: %v", err)
	}
	commId := initResp.CommId
	defer client.CommDestroy(ctx, &pb.CommDestroyRequest{CommId: commId})

	// Step 3: Initialize MLP and Load MNIST Data
	mlp := NewMLP()

	trainImagesPath := "data/train-images-idx3-ubyte.gz"
	trainLabelsPath := "data/train-labels-idx1-ubyte.gz"
	testImagesPath := "data/t10k-images-idx3-ubyte.gz"
	testLabelsPath := "data/t10k-labels-idx1-ubyte.gz"

	trainImages, numTrainImages := LoadMNISTImages(trainImagesPath)
	trainLabels, numTrainLabels := LoadMNISTLabels(trainLabelsPath)

	if numTrainImages != numTrainLabels {
		log.Fatalf("Train images and labels count mismatch: %d vs %d", numTrainImages, numTrainLabels)
	}

	fmt.Printf("Loaded %d training samples\n", numTrainImages)

	testImages, numTestImages := LoadMNISTImages(testImagesPath)
	testLabels, numTestLabels := LoadMNISTLabels(testLabelsPath)

	if numTestImages != numTestLabels {
		log.Fatalf("Test images and labels count mismatch: %d vs %d", numTestImages, numTestLabels)
	}

	fmt.Printf("Loaded %d test samples\n", numTestImages)

	trainData := trainImages
	trainSetLabels := trainLabels

	// Step 4: Distribute initial weights to devices
	for i, dev := range deviceClients {
		copyWeightsToDevice(ctx, dev, uint64(i+1), mlp)
	}

	// Step 5: Create a progress bar for epochs
	// Using OptionClearOnFinish to ensure each epoch's progress bar occupies the same line
	log.Println("Starting MLP training...")
	for epoch := 1; epoch <= numEpochs; epoch++ {
		// Declare totalBatches before using it in progressbar
		totalBatchesInt := len(trainData) / (batchSize * inputSize)

		// Create a progress bar for the current epoch's batches
		bar := progressbar.NewOptions(totalBatchesInt,
			progressbar.OptionSetDescription(fmt.Sprintf("Epoch %d/%d - Training Batches", epoch, numEpochs)),
			progressbar.OptionShowCount(),
			progressbar.OptionThrottle(65*time.Millisecond),
			progressbar.OptionClearOnFinish(),
			progressbar.OptionShowIts(),
		)

		var totalLoss float32
		var totalCorrect int
		var totalSamples int = totalBatchesInt * batchSize

		for batch := 0; batch < totalBatchesInt; batch++ {
			start := batch * batchSize * inputSize
			end := start + batchSize*inputSize
			batchData := trainData[start:end]

			labelStart := batch * batchSize * outputSize
			labelEnd := labelStart + batchSize*outputSize
			batchLabels := trainSetLabels[labelStart:labelEnd]

			logits, hidden, z1 := forwardPass(mlp, batchData)
			dW1, dB1, dW2, dB2, loss := backwardPass(mlp, batchData, hidden, z1, logits, batchLabels)
			totalLoss += loss

			// Compute accuracy for this batch
			batchLogits := make([]float32, len(logits))
			copy(batchLogits, logits)
			for b := 0; b < batchSize; b++ {
				sub := batchLogits[b*outputSize : b*outputSize+outputSize]
				softmax(sub)
			}
			correct := computeAccuracy(batchLogits, batchLabels)
			totalCorrect += correct

			allGrads := append(dW1, append(dB1, append(dW2, dB2...)...)...)
			copyGradientsToDevice(ctx, deviceClients, allGrads)

			_, err := client.AllReduceRing(ctx, &pb.AllReduceRingRequest{
				CommId: commId,
				Count:  uint64(len(allGrads) * 4),
			})
			if err != nil {
				log.Fatalf("AllReduceRing failed: %v", err)
			}

			aggGrads := retrieveGradientsFromDevice(ctx, deviceClients, len(allGrads))
			offset := 0
			dW1Agg := aggGrads[offset : offset+len(dW1)]
			offset += len(dW1)
			dB1Agg := aggGrads[offset : offset+len(dB1)]
			offset += len(dB1)
			dW2Agg := aggGrads[offset : offset+len(dW2)]
			offset += len(dW2)
			dB2Agg := aggGrads[offset : offset+len(dB2)]

			updateWeights(mlp, dW1Agg, dB1Agg, dW2Agg, dB2Agg)

			for i, dev := range deviceClients {
				copyWeightsToDevice(ctx, dev, uint64(i+1), mlp)
			}

			bar.Add(1)
		}

		bar.Finish()
		avgLoss := totalLoss / float32(totalBatchesInt)
		accuracy := float32(totalCorrect) / float32(totalSamples) * 100.0
		log.Printf("Epoch %d complete: Avg Loss: %.4f, Accuracy: %.2f%%", epoch, avgLoss, accuracy)
	}

	log.Println("Training complete.")

	// Test the model on test dataset with progress bar
	testModel(mlp, testImages, testLabels)
}
