package gpu_device_service

import (
	"context"
	"io"
	"log"
	"sync"

	pb "example.com/dsml/proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type StreamState struct {
	sendBuffAddr  uint64
	recvBuffAddr  uint64
	count         uint64
	srcRank       uint32
	dstRank       uint32
	status        pb.Status
	dataBuffer    []byte
	initiatedSend bool
	initiatedRecv bool
}

type GPUDeviceServer struct {
	pb.UnimplementedGPUDeviceServer

	mu           sync.Mutex
	deviceId     uint64
	memory       map[uint64][]byte
	memSize      int
	streams      map[uint64]*StreamState
	nextStreamId uint64
	minAddr      uint64
	maxAddr      uint64
}

func NewGPUDeviceServer(deviceId uint64, memSize int) *GPUDeviceServer {
	return &GPUDeviceServer{
		deviceId:     deviceId,
		memSize:      memSize,
		memory:       make(map[uint64][]byte),
		streams:      make(map[uint64]*StreamState),
		minAddr:      0x1000,
		maxAddr:      0x1000 + uint64(memSize),
		nextStreamId: 1,
	}
}

func (s *GPUDeviceServer) GetDeviceMetadata(ctx context.Context, req *pb.GetDeviceMetadataRequest) (*pb.GetDeviceMetadataResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	return &pb.GetDeviceMetadataResponse{
		Metadata: &pb.DeviceMetadata{
			DeviceId:   &pb.DeviceId{Value: s.deviceId},
			MinMemAddr: &pb.MemAddr{Value: s.minAddr},
			MaxMemAddr: &pb.MemAddr{Value: s.maxAddr},
		},
	}, nil
}

func (s *GPUDeviceServer) BeginSend(ctx context.Context, req *pb.BeginSendRequest) (*pb.BeginSendResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	streamId := s.nextStreamId
	s.nextStreamId++

	// No address bounds check for sendBuffAddr since it represents source on device;
	// Assume data is already on device. If needed, you can add similar checks.

	st := &StreamState{
		sendBuffAddr:  req.SendBuffAddr.Value,
		count:         req.NumBytes,
		dstRank:       req.DstRank.Value,
		status:        pb.Status_IN_PROGRESS,
		initiatedSend: true,
	}

	s.streams[streamId] = st

	return &pb.BeginSendResponse{
		Initiated: true,
		StreamId:  &pb.StreamId{Value: streamId},
	}, nil
}

func (s *GPUDeviceServer) BeginReceive(ctx context.Context, req *pb.BeginReceiveRequest) (*pb.BeginReceiveResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	str, ok := s.streams[req.StreamId.Value]
	if !ok {
		return &pb.BeginReceiveResponse{Initiated: false}, status.Errorf(codes.NotFound, "unknown streamId %d", req.StreamId.Value)
	}

	// Check address bounds for the receive buffer
	if req.RecvBuffAddr.Value < s.minAddr || req.RecvBuffAddr.Value >= s.maxAddr {
		return &pb.BeginReceiveResponse{Initiated: false},
			status.Errorf(codes.InvalidArgument, "RecvBuffAddr 0x%x out of bounds", req.RecvBuffAddr.Value)
	}

	str.recvBuffAddr = req.RecvBuffAddr.Value
	str.srcRank = req.SrcRank.Value
	str.initiatedRecv = true

	return &pb.BeginReceiveResponse{Initiated: true}, nil
}

func (s *GPUDeviceServer) StreamSend(server pb.GPUDevice_StreamSendServer) error {
	firstChunk, err := server.Recv()
	if err == io.EOF {
		// No data received
		return server.SendAndClose(&pb.StreamSendResponse{Success: false})
	}
	if err != nil {
		return err
	}

	streamId := firstChunk.GetStreamId()
	if streamId == 0 {
		return status.Errorf(codes.InvalidArgument, "no streamId in the first chunk")
	}

	s.mu.Lock()
	str, ok := s.streams[streamId]
	if !ok {
		s.mu.Unlock()
		return server.SendAndClose(&pb.StreamSendResponse{Success: false})
	}

	if !str.initiatedSend {
		str.status = pb.Status_FAILED
		s.mu.Unlock()
		log.Printf("Stream %d: send not initiated", streamId)
		return server.SendAndClose(&pb.StreamSendResponse{Success: false})
	}

	str.dataBuffer = append(str.dataBuffer, firstChunk.Data...)
	s.mu.Unlock()

	// Receive subsequent chunks
	for {
		chunk, err := server.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			s.mu.Lock()
			str.status = pb.Status_FAILED
			s.mu.Unlock()
			log.Printf("Stream %d: error receiving chunks: %v", streamId, err)
			return err
		}
		s.mu.Lock()
		str.dataBuffer = append(str.dataBuffer, chunk.Data...)
		s.mu.Unlock()
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if uint64(len(str.dataBuffer)) != str.count {
		str.status = pb.Status_FAILED
		log.Printf("Stream %d: data size mismatch. Expected %d, got %d", streamId, str.count, len(str.dataBuffer))
		return server.SendAndClose(&pb.StreamSendResponse{Success: false})
	}

	if !str.initiatedRecv {
		str.status = pb.Status_FAILED
		log.Printf("Stream %d: receive not initiated before data arrived", streamId)
		return server.SendAndClose(&pb.StreamSendResponse{Success: false})
	}

	// Write received data into device memory
	s.memory[str.recvBuffAddr] = append([]byte(nil), str.dataBuffer...)
	str.status = pb.Status_SUCCESS
	return server.SendAndClose(&pb.StreamSendResponse{Success: true})
}

func (s *GPUDeviceServer) GetStreamStatus(ctx context.Context, req *pb.GetStreamStatusRequest) (*pb.GetStreamStatusResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	str, ok := s.streams[req.StreamId.Value]
	if !ok {
		return &pb.GetStreamStatusResponse{Status: pb.Status_FAILED}, nil
	}

	return &pb.GetStreamStatusResponse{Status: str.status}, nil
}

func (s *GPUDeviceServer) Memcpy(ctx context.Context, req *pb.MemcpyRequest) (*pb.MemcpyResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if h2d := req.GetHostToDevice(); h2d != nil {
		addr := h2d.DstMemAddr.Value
		if addr < s.minAddr || addr >= s.maxAddr {
			return nil, status.Errorf(codes.InvalidArgument, "HostToDevice: address 0x%x out of bounds", addr)
		}
		s.memory[addr] = append([]byte(nil), h2d.HostSrcData...)
		return &pb.MemcpyResponse{
			Either: &pb.MemcpyResponse_HostToDevice{
				HostToDevice: &pb.MemcpyHostToDeviceResponse{Success: true},
			},
		}, nil
	}

	if d2h := req.GetDeviceToHost(); d2h != nil {
		addr := d2h.SrcMemAddr.Value
		if addr < s.minAddr || addr >= s.maxAddr {
			return nil, status.Errorf(codes.InvalidArgument, "DeviceToHost: address 0x%x out of bounds", addr)
		}
		data, ok := s.memory[addr]
		if !ok {
			return nil, status.Errorf(codes.Internal, "DeviceToHost: no data at address 0x%x", addr)
		}
		retData := append([]byte(nil), data...)
		return &pb.MemcpyResponse{
			Either: &pb.MemcpyResponse_DeviceToHost{
				DeviceToHost: &pb.MemcpyDeviceToHostResponse{DstData: retData},
			},
		}, nil
	}

	return nil, status.Errorf(codes.InvalidArgument, "Invalid Memcpy request: specify hostToDevice or deviceToHost")
}
