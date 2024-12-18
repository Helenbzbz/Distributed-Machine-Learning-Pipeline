package gpu_coordinator_service_test

import (
	"context"
	"net"
	"testing"
	"time"

	"example.com/dsml/gpu_coordinator_service"
	"example.com/dsml/gpu_device_service"
	pb "example.com/dsml/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// startGPUDeviceServer starts a GPUDeviceServer on a random port.
func startGPUDeviceServerWithLogging(t *testing.T, deviceID uint64, memSize int) (addr string, stopFunc func()) {
	lis, err := net.Listen("tcp", "localhost:0") // Use an ephemeral port
	if err != nil {
		t.Fatalf("failed to create listener: %v", err)
	}

	grpcServer := grpc.NewServer()
	deviceServer := gpu_device_service.NewGPUDeviceServer(deviceID, memSize)
	pb.RegisterGPUDeviceServer(grpcServer, deviceServer)

	go func() {
		if err := grpcServer.Serve(lis); err != nil {
			t.Logf("GPUDeviceServer error: %v", err)
		}
	}()

	return lis.Addr().String(), func() {
		grpcServer.Stop()
	}
}

// startCoordinatorServer starts the GPUCoordinatorServer on a random port.
func startCoordinatorServer(t *testing.T) (*gpu_coordinator_service.GPUCoordinatorServer, string, func()) {
	t.Helper()

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("failed to create listener: %v", err)
	}

	grpcServer := grpc.NewServer()
	coordServer := gpu_coordinator_service.NewGPUCoordinatorServer()
	pb.RegisterGPUCoordinatorServer(grpcServer, coordServer)

	go func() {
		if err := grpcServer.Serve(lis); err != nil {
			t.Logf("CoordinatorServer error: %v", err)
		}
	}()

	return coordServer, lis.Addr().String(), func() {
		coordServer.Stop()
		grpcServer.Stop()
	}
}

// Test CommInit
func TestCommInitWithInvalidDevices(t *testing.T) {
	// Start only 1 GPU device server
	addr, stopFn := startGPUDeviceServerWithLogging(t, uint64(1), 0x1000)
	defer stopFn()

	// Start coordinator
	_, coordAddr, coordStop := startCoordinatorServer(t)
	defer coordStop()

	// Connect to coordinator
	conn, err := grpc.Dial(coordAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("failed to dial coordinator: %v", err)
	}
	defer conn.Close()
	coordClient := pb.NewGPUCoordinatorClient(conn)

	// Invalid addresses (one valid, two invalid)
	req := &pb.CommInitRequest{
		NumDevices:      3,
		DeviceAddresses: []string{addr, "localhost:99999", "wrongformat"},
	}

	resp, err := coordClient.CommInit(context.Background(), req)
	if err == nil {
		t.Fatalf("expected error due to invalid device addresses, got success: %v", resp)
	}

	s, ok := status.FromError(err)
	if !ok || s.Code() != codes.Internal {
		t.Fatalf("expected Internal error, got %v", err)
	}
}

// Test Memcpy
func TestMemcpyHostToDeviceAndDeviceToHost(t *testing.T) {
	// Start a GPU device
	addr, stopFn := startGPUDeviceServerWithLogging(t, 1, 0x1000)
	defer stopFn()

	// Start coordinator
	_, coordAddr, coordStop := startCoordinatorServer(t)
	defer coordStop()

	// Connect to coordinator
	conn, err := grpc.Dial(coordAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("failed to dial coordinator: %v", err)
	}
	defer conn.Close()
	coordClient := pb.NewGPUCoordinatorClient(conn)

	// Init Comm with one device
	req := &pb.CommInitRequest{
		NumDevices:      1,
		DeviceAddresses: []string{addr},
	}
	resp, err := coordClient.CommInit(context.Background(), req)
	if err != nil || !resp.Success {
		t.Fatalf("CommInit failed: %v", err)
	}
	commId := resp.CommId

	// Host to Device Memcpy
	hostData := []byte("Hello GPU!")
	h2dReq := &pb.MemcpyRequest{
		Either: &pb.MemcpyRequest_HostToDevice{
			HostToDevice: &pb.MemcpyHostToDeviceRequest{
				HostSrcData: hostData,
				DstDeviceId: &pb.DeviceId{Value: resp.Devices[0].DeviceId.Value},
				DstMemAddr:  &pb.MemAddr{Value: 0x1000},
			},
		},
	}
	h2dResp, err := coordClient.Memcpy(context.Background(), h2dReq)
	if err != nil {
		t.Fatalf("HostToDevice Memcpy failed: %v", err)
	}
	if !h2dResp.GetHostToDevice().Success {
		t.Fatalf("HostToDevice Memcpy not successful")
	}

	// Device to Host Memcpy
	d2hReq := &pb.MemcpyRequest{
		Either: &pb.MemcpyRequest_DeviceToHost{
			DeviceToHost: &pb.MemcpyDeviceToHostRequest{
				SrcDeviceId: &pb.DeviceId{Value: resp.Devices[0].DeviceId.Value},
				SrcMemAddr:  &pb.MemAddr{Value: 0x1000},
				NumBytes:    uint64(len(hostData)),
			},
		},
	}

	d2hResp, err := coordClient.Memcpy(context.Background(), d2hReq)
	if err != nil {
		t.Fatalf("DeviceToHost Memcpy failed: %v", err)
	}
	if string(d2hResp.GetDeviceToHost().DstData) != string(hostData) {
		t.Errorf("expected '%s', got '%s'", string(hostData), string(d2hResp.GetDeviceToHost().DstData))
	}

	// Cleanup communicator
	destroyResp, err := coordClient.CommDestroy(context.Background(), &pb.CommDestroyRequest{CommId: commId})
	if err != nil || !destroyResp.Success {
		t.Fatalf("CommDestroy failed: %v", err)
	}
}

// Test GroupStart
func TestGroupOperationsWithoutComm(t *testing.T) {
	_, coordAddr, coordStop := startCoordinatorServer(t)
	defer coordStop()

	conn, err := grpc.Dial(coordAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("failed to connect coordinator: %v", err)
	}
	defer conn.Close()
	coordClient := pb.NewGPUCoordinatorClient(conn)

	nonExistentCommId := uint64(9999)

	// GroupStart on non-existent communicator
	_, err = coordClient.GroupStart(context.Background(), &pb.GroupStartRequest{CommId: nonExistentCommId})
	if err == nil {
		t.Fatal("expected error for GroupStart on non-existent communicator")
	}

	// GroupEnd on non-existent communicator
	_, err = coordClient.GroupEnd(context.Background(), &pb.GroupEndRequest{CommId: nonExistentCommId})
	if err == nil {
		t.Fatal("expected error for GroupEnd on non-existent communicator")
	}
}

// Test Comm Destroy
func TestCommDestroyInvalidId(t *testing.T) {
	_, coordAddr, coordStop := startCoordinatorServer(t)
	defer coordStop()

	conn, err := grpc.Dial(coordAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("failed to connect coordinator: %v", err)
	}
	defer conn.Close()
	coordClient := pb.NewGPUCoordinatorClient(conn)

	invalidCommId := uint64(8888)
	_, err = coordClient.CommDestroy(context.Background(), &pb.CommDestroyRequest{CommId: invalidCommId})
	if err == nil {
		t.Fatal("expected error destroying a non-existent communicator")
	}

	s, ok := status.FromError(err)
	if !ok || s.Code() != codes.NotFound {
		t.Fatalf("expected NotFound error, got %v", err)
	}
}

// All Reduce Ring
func TestCoordinatorAllReduceRing(t *testing.T) {
	deviceCount := 3
	deviceAddrs := make([]string, deviceCount)
	stops := make([]func(), deviceCount)

	// Start GPU device servers
	for i := 0; i < deviceCount; i++ {
		addr, stopFn := startGPUDeviceServerWithLogging(t, uint64(i+1), 0x1000)
		deviceAddrs[i] = addr
		stops[i] = stopFn
	}
	defer func() {
		for _, s := range stops {
			if s != nil {
				s()
			}
		}
	}()

	// Start coordinator
	_, coordAddr, coordStop := startCoordinatorServer(t)
	defer coordStop()

	// Connect to coordinator
	conn, err := grpc.Dial(coordAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("failed to dial coordinator: %v", err)
	}
	defer conn.Close()
	coordClient := pb.NewGPUCoordinatorClient(conn)

	// Initialize communicator
	req := &pb.CommInitRequest{NumDevices: uint32(deviceCount)}
	resp, err := coordClient.CommInit(context.Background(), req)
	if err != nil {
		t.Fatalf("CommInit failed: %v", err)
	}
	if !resp.Success {
		t.Fatalf("CommInit should be successful but got %v", resp.Success)
	}
	commId := resp.CommId

	// GroupStart
	gstartResp, err := coordClient.GroupStart(context.Background(), &pb.GroupStartRequest{CommId: commId})
	if err != nil {
		t.Fatalf("GroupStart failed: %v", err)
	}
	if !gstartResp.Success {
		t.Fatalf("GroupStart should be successful but got %v", gstartResp.Success)
	}

	// AllReduceRing
	arResp, err := coordClient.AllReduceRing(context.Background(), &pb.AllReduceRingRequest{
		CommId: commId,
		Count:  16,
	})

	if err != nil {
		t.Fatalf("AllReduceRing failed: %v", err)
	}
	if !arResp.Success {
		t.Fatalf("AllReduceRing should be successful but got %v", arResp.Success)
	}

	// Check communicator status
	statusResp, err := coordClient.GetCommStatus(context.Background(), &pb.GetCommStatusRequest{CommId: commId})
	if err != nil {
		t.Fatalf("GetCommStatus failed: %v", err)
	}
	if statusResp.Status != pb.Status_SUCCESS {
		t.Errorf("expected CommStatus to be SUCCESS, got %v", statusResp.Status)
	}

	// GroupEnd
	gendResp, err := coordClient.GroupEnd(context.Background(), &pb.GroupEndRequest{CommId: commId})
	if err != nil {
		t.Fatalf("GroupEnd failed: %v", err)
	}
	if !gendResp.Success {
		t.Fatalf("GroupEnd should be successful but got %v", gendResp.Success)
	}

	// Destroy communicator
	destroyResp, err := coordClient.CommDestroy(context.Background(), &pb.CommDestroyRequest{CommId: commId})
	if err != nil {
		t.Fatalf("CommDestroy failed: %v", err)
	}
	if !destroyResp.Success {
		t.Fatalf("CommDestroy should be successful but got %v", destroyResp.Success)
	}
}

func TestAllReduceRingSingleDevice(t *testing.T) {
	// Start a single GPU device
	addr, stopFn := startGPUDeviceServerWithLogging(t, 1, 0x1000)
	defer stopFn()

	// Start coordinator
	_, coordAddr, coordStop := startCoordinatorServer(t)
	defer coordStop()

	conn, err := grpc.Dial(coordAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("failed to connect coordinator: %v", err)
	}
	defer conn.Close()
	coordClient := pb.NewGPUCoordinatorClient(conn)

	// CommInit with single device
	req := &pb.CommInitRequest{
		NumDevices:      1,
		DeviceAddresses: []string{addr},
	}
	resp, err := coordClient.CommInit(context.Background(), req)
	if err != nil || !resp.Success {
		t.Fatalf("CommInit failed for single device: %v", err)
	}
	commId := resp.CommId

	// AllReduceRing
	arResp, err := coordClient.AllReduceRing(context.Background(), &pb.AllReduceRingRequest{CommId: commId})
	if err != nil {
		t.Fatalf("AllReduceRing failed: %v", err)
	}
	if !arResp.Success {
		t.Fatalf("expected AllReduceRing to succeed immediately with single device")
	}

	statusResp, err := coordClient.GetCommStatus(context.Background(), &pb.GetCommStatusRequest{CommId: commId})
	if err != nil {
		t.Fatalf("GetCommStatus failed: %v", err)
	}
	if statusResp.Status != pb.Status_SUCCESS {
		t.Errorf("expected SUCCESS status for single device all-reduce, got %v", statusResp.Status)
	}

	// Destroy communicator
	destroyResp, err := coordClient.CommDestroy(context.Background(), &pb.CommDestroyRequest{CommId: commId})
	if err != nil || !destroyResp.Success {
		t.Fatalf("CommDestroy failed: %v", err)
	}
}

func TestCoordinatorDeviceFailure(t *testing.T) {
	// Start devices
	deviceCount := 2
	deviceAddrs := make([]string, deviceCount)
	stops := make([]func(), deviceCount)
	for i := 0; i < deviceCount; i++ {
		addr, stopFn := startGPUDeviceServerWithLogging(t, uint64(i+1), 0x1000)
		deviceAddrs[i] = addr
		stops[i] = stopFn
	}
	defer func() {
		for _, s := range stops {
			if s != nil {
				s()
			}
		}
	}()

	// Start coordinator
	_, coordAddr, coordStop := startCoordinatorServer(t)
	defer coordStop()

	// Connect to coordinator
	conn, err := grpc.Dial(coordAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("failed to dial coordinator: %v", err)
	}
	defer conn.Close()
	coordClient := pb.NewGPUCoordinatorClient(conn)

	// Init Comm with the actual device addresses
	req := &pb.CommInitRequest{
		NumDevices:      uint32(deviceCount),
		DeviceAddresses: deviceAddrs,
	}
	resp, err := coordClient.CommInit(context.Background(), req)
	if err != nil {
		t.Fatalf("CommInit failed: %v", err)
	}
	if !resp.Success {
		t.Fatalf("CommInit should be successful but got %v", resp.Success)
	}
	commId := resp.CommId

	// Simulate a device failure by stopping one device
	stops[0]()
	stops[0] = nil

	// Wait longer than healthInterval for the health check loop (5s by default)
	// We wait 6s to ensure at least one health check has run
	time.Sleep(6 * time.Second)

	statusResp, err := coordClient.GetCommStatus(context.Background(), &pb.GetCommStatusRequest{CommId: commId})
	if err != nil {
		t.Fatalf("GetCommStatus failed: %v", err)
	}
	if statusResp.Status != pb.Status_FAILED {
		t.Errorf("expected CommStatus to be FAILED after device failure, got %v", statusResp.Status)
	}
}
