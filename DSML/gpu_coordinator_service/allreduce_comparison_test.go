package gpu_coordinator_service_test

import (
	"context"
	"net"
	"testing"
	"time"

	"example.com/dsml/gpu_device_service"
	pb "example.com/dsml/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Utility: Start GPU Device Server
func startGPUDeviceServer(t *testing.T, deviceID uint64, memSize int) (string, func()) {
	lis, err := net.Listen("tcp", "localhost:0") // Random available port
	if err != nil {
		t.Fatalf("Failed to create listener: %v", err)
	}

	grpcServer := grpc.NewServer()
	deviceServer := gpu_device_service.NewGPUDeviceServer(deviceID, memSize)
	pb.RegisterGPUDeviceServer(grpcServer, deviceServer)

	go func() { grpcServer.Serve(lis) }()
	return lis.Addr().String(), func() { grpcServer.Stop() }
}

// Test: Compare Naive and Ring AllReduce
func TestAllReduceComparison(t *testing.T) {
	// Start GPU devices
	deviceCount := 3
	deviceAddrs := make([]string, deviceCount)
	stops := make([]func(), deviceCount)

	for i := 0; i < deviceCount; i++ {
		addr, stop := startGPUDeviceServer(t, uint64(i+1), 0x2000)
		deviceAddrs[i] = addr
		stops[i] = stop
	}
	defer func() {
		for _, stop := range stops {
			stop()
		}
	}()

	// Start Coordinator
	_, coordAddr, stopCoord := startCoordinatorServer(t)
	defer stopCoord()

	// Connect to Coordinator
	conn, err := grpc.Dial(coordAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to connect to coordinator: %v", err)
	}
	defer conn.Close()
	coordClient := pb.NewGPUCoordinatorClient(conn)

	ctx := context.Background()

	// Initialize Communicator
	initResp, err := coordClient.CommInit(ctx, &pb.CommInitRequest{
		NumDevices:      uint32(deviceCount),
		DeviceAddresses: deviceAddrs,
	})
	if err != nil {
		t.Fatalf("CommInit failed: %v", err)
	}
	if len(initResp.Devices) != deviceCount {
		t.Fatalf("CommInit succeeded but returned %d devices, expected %d", len(initResp.Devices), deviceCount)
	}
	commId := initResp.CommId
	t.Logf("CommInit successful: CommId=%d, Devices=%d", commId, len(initResp.Devices))

	// ** New Code: Initialize device memory with data **
	dataSize := uint64(1024 * 1024) // 1 MB
	initialData := make([]byte, dataSize)
	// Fill initialData with some pattern, e.g., all ones
	for i := range initialData {
		initialData[i] = 1
	}

	// Write initial data to each device at 0x1000
	for _, dev := range initResp.Devices {
		_, err := coordClient.Memcpy(ctx, &pb.MemcpyRequest{
			Either: &pb.MemcpyRequest_HostToDevice{
				HostToDevice: &pb.MemcpyHostToDeviceRequest{
					HostSrcData: initialData,
					DstDeviceId: dev.DeviceId,
					DstMemAddr:  &pb.MemAddr{Value: 0x1000},
				},
			},
		})
		if err != nil {
			t.Fatalf("Failed to initialize device memory: %v", err)
		}
	}

	latencyMs := uint32(10)

	// Naive AllReduce
	naiveStart := time.Now()
	naiveResp, err := coordClient.NaiveAllReduce(ctx, &pb.NaiveAllReduceRequest{
		CommId:    commId,
		DataSize:  dataSize,
		LatencyMs: latencyMs,
	})
	naiveElapsed := time.Since(naiveStart).Milliseconds()
	if err != nil || !naiveResp.Success {
		t.Fatalf("NaiveAllReduce failed: %v", err)
	}

	// Ring AllReduce
	ringStart := time.Now()
	ringResp, err := coordClient.AllReduceRing(ctx, &pb.AllReduceRingRequest{
		CommId: commId,
		Count:  dataSize / 4,
	})
	ringElapsed := time.Since(ringStart).Milliseconds()
	if err != nil || !ringResp.Success {
		t.Fatalf("AllReduceRing failed: %v", err)
	}

	// Results
	t.Logf("\n--- AllReduce Comparison ---")
	t.Logf("Naive AllReduce: Time=%dms, DataTransferred=%d bytes", naiveElapsed, naiveResp.TotalDataTransferred)
	t.Logf("Ring AllReduce:  Time=%dms", ringElapsed)

	// Cleanup
	_, _ = coordClient.CommDestroy(ctx, &pb.CommDestroyRequest{CommId: commId})
}
