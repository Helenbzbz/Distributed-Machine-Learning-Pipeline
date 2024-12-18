package gpu_coordinator_service

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	pb "example.com/dsml/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

type CommStatus int

const (
	CommStatus_IN_PROGRESS CommStatus = iota
	CommStatus_SUCCESS
	CommStatus_FAILED
)

type DeviceInfo struct {
	Rank       uint32
	DeviceId   uint64
	Addr       string
	Client     pb.GPUDeviceClient
	LastHealth time.Time
}

type Communicator struct {
	Id          uint64
	Devices     []*DeviceInfo
	Status      CommStatus
	mu          sync.Mutex
	groupActive bool
}

type GPUCoordinatorServer struct {
	pb.UnimplementedGPUCoordinatorServer

	mu             sync.Mutex
	nextCommId     uint64
	communicators  map[uint64]*Communicator
	healthInterval time.Duration
	stopHealthCh   chan struct{}

	// Simulated host memory: DeviceId -> MemAddr -> data bytes
	hostMemory map[uint64]map[uint64][]byte
}

func NewGPUCoordinatorServer() *GPUCoordinatorServer {
	s := &GPUCoordinatorServer{
		communicators:  make(map[uint64]*Communicator),
		healthInterval: 5 * time.Second,
		stopHealthCh:   make(chan struct{}),
		hostMemory:     make(map[uint64]map[uint64][]byte),
	}
	go s.healthCheckLoop()
	return s
}

func (s *GPUCoordinatorServer) Stop() {
	close(s.stopHealthCh)
}

func (s *GPUCoordinatorServer) healthCheckLoop() {
	ticker := time.NewTicker(s.healthInterval)
	defer ticker.Stop()
	for {
		select {
		case <-s.stopHealthCh:
			return
		case <-ticker.C:
			s.checkAllCommunicatorHealth()
		}
	}
}

func (s *GPUCoordinatorServer) checkAllCommunicatorHealth() {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, comm := range s.communicators {
		s.checkCommunicatorHealth(comm)
	}
}

func (s *GPUCoordinatorServer) checkCommunicatorHealth(comm *Communicator) {
	comm.mu.Lock()
	defer comm.mu.Unlock()

	if comm.Status == CommStatus_FAILED {
		return
	}

	var aliveDevices []*DeviceInfo
	var failedDevices []*DeviceInfo

	for _, d := range comm.Devices {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_, err := d.Client.GetDeviceMetadata(ctx, &pb.GetDeviceMetadataRequest{})
		cancel()
		if err != nil {
			log.Printf("Device %d at %s unreachable: %v", d.DeviceId, d.Addr, err)
			failedDevices = append(failedDevices, d)
		} else {
			aliveDevices = append(aliveDevices, d)
		}
	}

	comm.Devices = aliveDevices
	if len(failedDevices) > 0 {
		log.Printf("Communicator %d lost devices; marking as FAILED", comm.Id)
		comm.Status = CommStatus_FAILED
	}
}

func (s *GPUCoordinatorServer) CommInit(ctx context.Context, req *pb.CommInitRequest) (*pb.CommInitResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	commId := s.nextCommId
	s.nextCommId++

	var devices []*DeviceInfo
	var errors []string

	for i, addr := range req.GetDeviceAddresses() {
		var client pb.GPUDeviceClient
		var err error

		// Retry connection up to 3 times
		for attempt := 1; attempt <= 3; attempt++ {
			conn, connErr := grpc.Dial(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
			if connErr == nil {
				client = pb.NewGPUDeviceClient(conn)
				break
			}
			err = connErr
			log.Printf("Attempt %d to connect to device at %s failed: %v", attempt, addr, err)
			time.Sleep(500 * time.Millisecond)
		}

		if err != nil {
			errors = append(errors, fmt.Sprintf("failed to connect to device at %s: %v", addr, err))
			continue
		}

		mdResp, err := client.GetDeviceMetadata(ctx, &pb.GetDeviceMetadataRequest{})
		if err != nil {
			errors = append(errors, fmt.Sprintf("GetDeviceMetadata failed for %s: %v", addr, err))
			continue
		}

		devices = append(devices, &DeviceInfo{
			Rank:       uint32(i),
			DeviceId:   mdResp.Metadata.DeviceId.Value,
			Addr:       addr,
			Client:     client,
			LastHealth: time.Now(),
		})
	}

	if len(errors) > 0 {
		return nil, status.Errorf(codes.Internal, "CommInit encountered errors: %v", errors)
	}

	comm := &Communicator{
		Id:      commId,
		Devices: devices,
		Status:  CommStatus_IN_PROGRESS,
	}
	s.communicators[commId] = comm

	var devMetas []*pb.DeviceMetadata
	for _, d := range devices {
		devMetas = append(devMetas, &pb.DeviceMetadata{
			DeviceId:   &pb.DeviceId{Value: d.DeviceId},
			MinMemAddr: &pb.MemAddr{Value: 0x1000},
			MaxMemAddr: &pb.MemAddr{Value: 0x2000},
		})
	}

	return &pb.CommInitResponse{
		Success: true,
		CommId:  commId,
		Devices: devMetas,
	}, nil
}

func (s *GPUCoordinatorServer) Memcpy(ctx context.Context, req *pb.MemcpyRequest) (*pb.MemcpyResponse, error) {
	if h2d := req.GetHostToDevice(); h2d != nil {
		did := h2d.DstDeviceId.Value
		addr := h2d.DstMemAddr.Value

		if addr < 0x1000 || addr > 0x2000 {
			return nil, status.Errorf(codes.InvalidArgument, "Memory address out of bounds")
		}

		s.mu.Lock()
		if _, ok := s.hostMemory[did]; !ok {
			s.hostMemory[did] = make(map[uint64][]byte)
		}
		s.hostMemory[did][addr] = append([]byte(nil), h2d.HostSrcData...)
		s.mu.Unlock()

		return &pb.MemcpyResponse{
			Either: &pb.MemcpyResponse_HostToDevice{
				HostToDevice: &pb.MemcpyHostToDeviceResponse{Success: true},
			},
		}, nil
	} else if d2h := req.GetDeviceToHost(); d2h != nil {
		did := d2h.SrcDeviceId.Value
		addr := d2h.SrcMemAddr.Value

		s.mu.Lock()
		// Ensure device map is initialized before reading
		if _, ok := s.hostMemory[did]; !ok {
			// If there's no entry, this means no data was ever written. Return error.
			s.mu.Unlock()
			return nil, status.Errorf(codes.Internal, "No memory initialized for device %d", did)
		}
		data, ok := s.hostMemory[did][addr]
		s.mu.Unlock()
		if !ok {
			return nil, status.Errorf(codes.Internal, "DeviceToHost: no data at address 0x%x", addr)
		}

		return &pb.MemcpyResponse{
			Either: &pb.MemcpyResponse_DeviceToHost{
				DeviceToHost: &pb.MemcpyDeviceToHostResponse{DstData: data},
			},
		}, nil
	}

	return nil, status.Errorf(codes.InvalidArgument, "Invalid Memcpy request: must specify hostToDevice or deviceToHost")
}

func (s *GPUCoordinatorServer) GroupStart(ctx context.Context, req *pb.GroupStartRequest) (*pb.GroupStartResponse, error) {
	s.mu.Lock()
	comm, ok := s.communicators[req.CommId]
	s.mu.Unlock()
	if !ok {
		return nil, status.Errorf(codes.NotFound, "Communicator not found")
	}

	comm.mu.Lock()
	comm.groupActive = true
	comm.mu.Unlock()

	return &pb.GroupStartResponse{Success: true}, nil
}

func (s *GPUCoordinatorServer) GroupEnd(ctx context.Context, req *pb.GroupEndRequest) (*pb.GroupEndResponse, error) {
	s.mu.Lock()
	comm, ok := s.communicators[req.CommId]
	s.mu.Unlock()
	if !ok {
		return nil, status.Errorf(codes.NotFound, "Communicator not found")
	}

	comm.mu.Lock()
	comm.groupActive = false
	comm.mu.Unlock()

	return &pb.GroupEndResponse{Success: true}, nil
}

// AllReduceRing performs a real ring all-reduce using a scatter-reduce + allgather algorithm.
func (s *GPUCoordinatorServer) AllReduceRing(ctx context.Context, req *pb.AllReduceRingRequest) (*pb.AllReduceRingResponse, error) {
	s.mu.Lock()
	comm, ok := s.communicators[req.CommId]
	s.mu.Unlock()
	if !ok {
		return nil, status.Errorf(codes.NotFound, "Communicator not found")
	}

	comm.mu.Lock()
	if comm.Status == CommStatus_FAILED {
		comm.mu.Unlock()
		return nil, status.Errorf(codes.FailedPrecondition, "Communicator is in FAILED state")
	}
	n := len(comm.Devices)
	comm.mu.Unlock()

	if n < 2 {
		// With only one device, no reduction is needed
		comm.mu.Lock()
		comm.Status = CommStatus_SUCCESS
		comm.mu.Unlock()
		return &pb.AllReduceRingResponse{Success: true}, nil
	}

	originalTotalBytes := req.Count
	totalBytes := originalTotalBytes
	remainder := totalBytes % uint64(n)
	if remainder != 0 {
		// Pad data so that totalBytes is divisible by n
		padding := uint64(n) - remainder
		newTotalBytes := totalBytes + padding

		// Pad each device's memory at 0x1000 to newTotalBytes with zeros
		s.mu.Lock()
		for _, d := range comm.Devices {
			// Ensure hostMemory map for this device is initialized
			if _, ok := s.hostMemory[d.DeviceId]; !ok {
				s.hostMemory[d.DeviceId] = make(map[uint64][]byte)
			}

			data, ok := s.hostMemory[d.DeviceId][0x1000]
			if !ok {
				// If there's no data yet at 0x1000, initialize it to an empty slice
				data = []byte{}
			}

			// Ensure we have at least originalTotalBytes in data
			if uint64(len(data)) < originalTotalBytes {
				extended := make([]byte, originalTotalBytes)
				copy(extended, data)
				data = extended
			}

			extended := make([]byte, newTotalBytes)
			copy(extended, data)
			// Assign back to hostMemory
			s.hostMemory[d.DeviceId][0x1000] = extended
		}
		s.mu.Unlock()

		totalBytes = newTotalBytes
	}

	segmentSize := totalBytes / uint64(n)

	// Scatter-Reduce phase
	for step := 0; step < n-1; step++ {
		if err := s.allReduceRingStep(ctx, comm, n, totalBytes, segmentSize, step, true); err != nil {
			comm.mu.Lock()
			comm.Status = CommStatus_FAILED
			comm.mu.Unlock()
			return nil, err
		}
	}

	// Allgather phase
	for step := 0; step < n-1; step++ {
		if err := s.allReduceRingStep(ctx, comm, n, totalBytes, segmentSize, step, false); err != nil {
			comm.mu.Lock()
			comm.Status = CommStatus_FAILED
			comm.mu.Unlock()
			return nil, err
		}
	}

	// Truncate arrays back to the original size if we padded them
	if remainder != 0 {
		s.mu.Lock()
		for _, d := range comm.Devices {
			data := s.hostMemory[d.DeviceId][0x1000]
			if uint64(len(data)) > originalTotalBytes {
				truncated := data[:originalTotalBytes]
				s.hostMemory[d.DeviceId][0x1000] = truncated
			}
		}
		s.mu.Unlock()
	}

	comm.mu.Lock()
	comm.Status = CommStatus_SUCCESS
	comm.mu.Unlock()

	return &pb.AllReduceRingResponse{Success: true}, nil
}

// allReduceRingStep performs one step of either the scatter-reduce or allgather phase of the ring all-reduce.
func (s *GPUCoordinatorServer) allReduceRingStep(ctx context.Context, comm *Communicator, n int, totalBytes, segmentSize uint64, step int, scatterReduce bool) error {
	comm.mu.Lock()
	devices := comm.Devices
	comm.mu.Unlock()

	errCh := make(chan error, n)
	var wg sync.WaitGroup

	for _, device := range devices {
		wg.Add(1)
		go func(d *DeviceInfo) {
			defer wg.Done()

			rank := d.Rank
			// Compute sendIndex and recvIndex for this step
			sendIndex := (int(rank) - step) % n
			if sendIndex < 0 {
				sendIndex += n
			}
			recvIndex := (int(rank) - step - 1) % n
			if recvIndex < 0 {
				recvIndex += n
			}

			sendOffset := uint64(sendIndex) * segmentSize
			recvOffset := uint64(recvIndex) * segmentSize

			// Identify next and prev ranks in the ring
			nextRank := (rank + 1) % uint32(n)
			prevRank := (rank + uint32(n) - 1) % uint32(n)

			// Find devices for nextRank and prevRank
			var nextDev, prevDev *DeviceInfo
			for _, dd := range devices {
				if dd.Rank == nextRank {
					nextDev = dd
				}
				if dd.Rank == prevRank {
					prevDev = dd
				}
			}

			if nextDev == nil || prevDev == nil {
				errCh <- fmt.Errorf("rank %d: missing neighbors in ring (nextDev=%v, prevDev=%v)", rank, nextDev, prevDev)
				return
			}

			// BeginSend
			sendResp, err := d.Client.BeginSend(ctx, &pb.BeginSendRequest{
				SendBuffAddr: &pb.MemAddr{Value: 0x1000},
				NumBytes:     segmentSize,
				DstRank:      &pb.Rank{Value: nextRank},
			})
			if err != nil {
				errCh <- fmt.Errorf("rank %d: BeginSend failed: %v", rank, err)
				return
			}

			// BeginReceive
			recvResp, err := d.Client.BeginReceive(ctx, &pb.BeginReceiveRequest{
				StreamId:     sendResp.StreamId,
				RecvBuffAddr: &pb.MemAddr{Value: 0x2000},
				NumBytes:     segmentSize,
				SrcRank:      &pb.Rank{Value: prevRank},
			})
			if err != nil || !recvResp.Initiated {
				errCh <- fmt.Errorf("rank %d: BeginReceive failed or not initiated: err=%v initiated=%v", rank, err, recvResp.Initiated)
				return
			}

			s.mu.Lock()
			data, okData := s.hostMemory[d.DeviceId][0x1000]
			s.mu.Unlock()

			if !okData {
				errCh <- fmt.Errorf("rank %d: no data found at device %d address 0x1000", rank, d.DeviceId)
				return
			}

			if uint64(len(data)) < sendOffset+segmentSize {
				errCh <- fmt.Errorf("rank %d: data too short: have %d bytes, need %d for send segment", rank, len(data), sendOffset+segmentSize)
				return
			}

			dataToSend := data[sendOffset : sendOffset+segmentSize]

			// StreamSend
			stream, err := d.Client.StreamSend(ctx)
			if err != nil {
				errCh <- fmt.Errorf("rank %d: StreamSend failed: %v", rank, err)
				return
			}

			if err = stream.Send(&pb.DataChunk{
				StreamId: sendResp.StreamId.Value,
				Data:     dataToSend,
			}); err != nil {
				errCh <- fmt.Errorf("rank %d: StreamSend data failed: %v", rank, err)
				return
			}

			if _, err = stream.CloseAndRecv(); err != nil {
				errCh <- fmt.Errorf("rank %d: StreamSend close failed: %v", rank, err)
				return
			}

			// Wait for successful receive on the device side
			for {
				statusResp, err := d.Client.GetStreamStatus(ctx, &pb.GetStreamStatusRequest{StreamId: sendResp.StreamId})
				if err != nil {
					errCh <- fmt.Errorf("rank %d: GetStreamStatus failed: %v", rank, err)
					return
				}
				if statusResp.Status == pb.Status_SUCCESS {
					break
				} else if statusResp.Status == pb.Status_FAILED {
					errCh <- fmt.Errorf("rank %d: stream failed", rank)
					return
				}
				time.Sleep(50 * time.Millisecond)
			}

			// Fetch received data from device memory to coordinator hostMemory
			d2hReq := &pb.MemcpyRequest{
				Either: &pb.MemcpyRequest_DeviceToHost{
					DeviceToHost: &pb.MemcpyDeviceToHostRequest{
						SrcDeviceId: &pb.DeviceId{Value: d.DeviceId},
						SrcMemAddr:  &pb.MemAddr{Value: 0x2000},
						NumBytes:    segmentSize,
					},
				},
			}

			resp, err := d.Client.Memcpy(ctx, d2hReq)
			if err != nil {
				errCh <- fmt.Errorf("rank %d: failed to Memcpy from device: %v", rank, err)
				return
			}

			s.mu.Lock()
			if _, ok := s.hostMemory[d.DeviceId]; !ok {
				s.hostMemory[d.DeviceId] = make(map[uint64][]byte)
			}
			s.hostMemory[d.DeviceId][0x2000] = resp.GetDeviceToHost().DstData
			recvData := s.hostMemory[d.DeviceId][0x2000]
			localData := s.hostMemory[d.DeviceId][0x1000]
			s.mu.Unlock()

			if uint64(len(recvData)) < segmentSize {
				errCh <- fmt.Errorf("rank %d: received data too short: have %d bytes, need %d", rank, len(recvData), segmentSize)
				return
			}

			// Ensure localData is long enough
			if uint64(len(localData)) < recvOffset+segmentSize {
				extended := make([]byte, recvOffset+segmentSize)
				copy(extended, localData)
				localData = extended
			}

			// Perform reduce or gather
			if scatterReduce {
				for i := uint64(0); i < segmentSize; i++ {
					localData[recvOffset+i] += recvData[i]
				}
			} else {
				copy(localData[recvOffset:recvOffset+segmentSize], recvData[:segmentSize])
			}

			s.mu.Lock()
			if _, ok := s.hostMemory[d.DeviceId]; !ok {
				s.hostMemory[d.DeviceId] = make(map[uint64][]byte)
			}
			s.hostMemory[d.DeviceId][0x1000] = localData
			s.mu.Unlock()

		}(device)
	}

	wg.Wait()
	select {
	case e := <-errCh:
		return e
	default:
		log.Printf("AllReduceRing step completed successfully for Communicator %d", comm.Id)
		return nil
	}
}

// GetCommStatus returns the current communicator status
func (s *GPUCoordinatorServer) GetCommStatus(ctx context.Context, req *pb.GetCommStatusRequest) (*pb.GetCommStatusResponse, error) {
	s.mu.Lock()
	comm, ok := s.communicators[req.CommId]
	s.mu.Unlock()

	if !ok {
		return nil, status.Errorf(codes.NotFound, "Communicator not found")
	}

	comm.mu.Lock()
	defer comm.mu.Unlock()

	var st pb.Status
	switch comm.Status {
	case CommStatus_IN_PROGRESS:
		st = pb.Status_IN_PROGRESS
	case CommStatus_SUCCESS:
		st = pb.Status_SUCCESS
	case CommStatus_FAILED:
		st = pb.Status_FAILED
	default:
		st = pb.Status_FAILED
	}

	return &pb.GetCommStatusResponse{Status: st}, nil
}

func (s *GPUCoordinatorServer) CommDestroy(ctx context.Context, req *pb.CommDestroyRequest) (*pb.CommDestroyResponse, error) {
	s.mu.Lock()
	comm, ok := s.communicators[req.CommId]
	if !ok {
		s.mu.Unlock()
		return nil, status.Errorf(codes.NotFound, "Communicator not found")
	}
	delete(s.communicators, req.CommId)
	s.mu.Unlock()

	log.Printf("Communicator %d destroyed", comm.Id)
	return &pb.CommDestroyResponse{Success: true}, nil
}

// NaiveAllReduce is a basic reference all-reduce method (gather + reduce + broadcast), not used for ring.
func (s *GPUCoordinatorServer) NaiveAllReduce(ctx context.Context, req *pb.NaiveAllReduceRequest) (*pb.NaiveAllReduceResponse, error) {
	s.mu.Lock()
	comm, ok := s.communicators[req.CommId]
	s.mu.Unlock()
	if !ok {
		return nil, status.Errorf(codes.NotFound, "Communicator not found")
	}

	comm.mu.Lock()
	if comm.Status == CommStatus_FAILED {
		comm.mu.Unlock()
		return nil, status.Errorf(codes.FailedPrecondition, "Communicator is in FAILED state")
	}
	devices := comm.Devices
	comm.mu.Unlock()

	dataSize := req.DataSize
	latency := req.LatencyMs

	if len(devices) == 0 {
		return nil, status.Errorf(codes.Internal, "No devices available in the communicator")
	}

	initialData := make([]byte, dataSize)
	for i := range initialData {
		initialData[i] = 1
	}

	ctx2 := context.Background()
	for _, device := range devices {
		time.Sleep(time.Duration(latency) * time.Millisecond)
		h2dReq := &pb.MemcpyRequest{
			Either: &pb.MemcpyRequest_HostToDevice{
				HostToDevice: &pb.MemcpyHostToDeviceRequest{
					HostSrcData: initialData,
					DstDeviceId: &pb.DeviceId{Value: device.DeviceId},
					DstMemAddr:  &pb.MemAddr{Value: 0x1000},
				},
			},
		}

		_, err := device.Client.Memcpy(ctx2, h2dReq)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "Failed to initialize data for device %d: %v", device.DeviceId, err)
		}
	}

	startTime := time.Now()
	gatheredData := make([][]byte, len(devices))
	totalDataTransferred := 0

	for i, device := range devices {
		time.Sleep(time.Duration(latency) * time.Millisecond)
		d2hReq := &pb.MemcpyRequest{
			Either: &pb.MemcpyRequest_DeviceToHost{
				DeviceToHost: &pb.MemcpyDeviceToHostRequest{
					SrcDeviceId: &pb.DeviceId{Value: device.DeviceId},
					SrcMemAddr:  &pb.MemAddr{Value: 0x1000},
					NumBytes:    dataSize,
				},
			},
		}
		resp, err := device.Client.Memcpy(ctx2, d2hReq)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "Failed to gather data from device %d: %v", device.DeviceId, err)
		}
		gatheredData[i] = resp.GetDeviceToHost().DstData
		totalDataTransferred += int(dataSize)
	}

	reducedData := make([]byte, dataSize)
	for _, data := range gatheredData {
		for j := 0; j < int(dataSize); j++ {
			reducedData[j] += data[j]
		}
	}

	for _, device := range devices {
		time.Sleep(time.Duration(latency) * time.Millisecond)
		h2dReq := &pb.MemcpyRequest{
			Either: &pb.MemcpyRequest_HostToDevice{
				HostToDevice: &pb.MemcpyHostToDeviceRequest{
					HostSrcData: reducedData,
					DstDeviceId: &pb.DeviceId{Value: device.DeviceId},
					DstMemAddr:  &pb.MemAddr{Value: 0x2000},
				},
			},
		}
		_, err := device.Client.Memcpy(ctx2, h2dReq)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "Failed to broadcast data to device %d: %v", device.DeviceId, err)
		}
		totalDataTransferred += int(dataSize)
	}

	endTime := time.Now()
	elapsedTime := endTime.Sub(startTime).Milliseconds()

	log.Printf("Naive AllReduce completed: DataSize=%d, Latency=%dms, TotalTime=%dms, TotalDataTransferred=%d bytes",
		dataSize, latency, elapsedTime, totalDataTransferred)

	return &pb.NaiveAllReduceResponse{
		Success:              true,
		TotalTimeMs:          int64(elapsedTime),
		TotalDataTransferred: int64(totalDataTransferred),
	}, nil
}
