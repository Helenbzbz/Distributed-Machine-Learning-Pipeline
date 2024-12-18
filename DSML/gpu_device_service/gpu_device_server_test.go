package gpu_device_service

import (
	"context"
	"io"
	"testing"

	pb "example.com/dsml/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Mock StreamSendServer Implementation
type mockStreamSendServer struct {
	grpc.ServerStream
	chunks []*pb.DataChunk
	i      int
	ctx    context.Context
}

func (m *mockStreamSendServer) Recv() (*pb.DataChunk, error) {
	if m.i >= len(m.chunks) {
		return nil, io.EOF
	}
	chunk := m.chunks[m.i]
	m.i++
	return chunk, nil
}

func (m *mockStreamSendServer) SendAndClose(resp *pb.StreamSendResponse) error {
	if !resp.Success {
		return status.Error(codes.Internal, "stream send failed")
	}
	return nil
}

func (m *mockStreamSendServer) Context() context.Context {
	return m.ctx
}

func (m *mockStreamSendServer) RecvMsg(v interface{}) error {
	return nil
}

func TestGetDeviceMetadata(t *testing.T) {
	server := NewGPUDeviceServer(1001, 1024*1024) // 1MB memory

	resp, err := server.GetDeviceMetadata(context.Background(), &pb.GetDeviceMetadataRequest{})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if resp.Metadata.DeviceId.Value != 1001 {
		t.Errorf("expected deviceId 1001, got %d", resp.Metadata.DeviceId.Value)
	}
	if resp.Metadata.MinMemAddr.Value != 0x1000 {
		t.Errorf("expected minMemAddr 0x1000, got 0x%x", resp.Metadata.MinMemAddr.Value)
	}
	if resp.Metadata.MaxMemAddr.Value != 0x101000 {
		t.Errorf("expected maxMemAddr 0x101000, got 0x%x", resp.Metadata.MaxMemAddr.Value)
	}
}

func TestBeginSend(t *testing.T) {
	server := NewGPUDeviceServer(1001, 1024*1024)

	resp, err := server.BeginSend(context.Background(), &pb.BeginSendRequest{
		SendBuffAddr: &pb.MemAddr{Value: 0x1000},
		NumBytes:     64,
		DstRank:      &pb.Rank{Value: 1},
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp.StreamId == nil || resp.StreamId.Value == 0 {
		t.Errorf("expected valid streamId, got %v", resp.StreamId)
	}
}

func TestBeginReceive(t *testing.T) {
	server := NewGPUDeviceServer(1001, 1024*1024)

	server.mu.Lock()
	server.streams[1] = &StreamState{
		sendBuffAddr: 0x1000,
		count:        64,
		dstRank:      1,
		status:       pb.Status_IN_PROGRESS,
	}
	server.mu.Unlock()

	resp, err := server.BeginReceive(context.Background(), &pb.BeginReceiveRequest{
		StreamId:     &pb.StreamId{Value: 1},
		RecvBuffAddr: &pb.MemAddr{Value: 0x2000},
		NumBytes:     64,
		SrcRank:      &pb.Rank{Value: 0},
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !resp.Initiated {
		t.Errorf("expected initiated true, got %v", resp.Initiated)
	}
}

func TestStreamSend(t *testing.T) {
	server := NewGPUDeviceServer(1001, 1024*1024)

	stream := &mockStreamSendServer{
		chunks: []*pb.DataChunk{
			{StreamId: 1, Data: []byte("chunk1")},
			{StreamId: 1, Data: []byte("chunk2")},
		},
		ctx: context.Background(),
	}

	server.mu.Lock()
	server.streams[1] = &StreamState{
		sendBuffAddr:  0x1000,
		recvBuffAddr:  0x2000,
		count:         12,
		srcRank:       0,
		dstRank:       1,
		status:        pb.Status_IN_PROGRESS,
		initiatedSend: true,
		initiatedRecv: true,
	}
	server.mu.Unlock()

	err := server.StreamSend(stream)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	server.mu.Lock()
	defer server.mu.Unlock()

	data := server.memory[0x2000]
	expected := []byte("chunk1chunk2")
	if string(data) != string(expected) {
		t.Errorf("expected %q, got %q", expected, data)
	}
}

func TestGetStreamStatus(t *testing.T) {
	server := NewGPUDeviceServer(1001, 1024*1024)

	server.mu.Lock()
	server.streams[1] = &StreamState{
		status: pb.Status_IN_PROGRESS,
	}
	server.mu.Unlock()

	resp, err := server.GetStreamStatus(context.Background(), &pb.GetStreamStatusRequest{
		StreamId: &pb.StreamId{Value: 1},
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp.Status != pb.Status_IN_PROGRESS {
		t.Errorf("expected status IN_PROGRESS, got %v", resp.Status)
	}
}
