package main

import (
	"log"
	"net"

	"example.com/dsml/gpu_coordinator_service"
	pb "example.com/dsml/proto"
	"google.golang.org/grpc"
)

func main() {
	lis, err := net.Listen("tcp", "localhost:50051")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	coordServer := gpu_coordinator_service.NewGPUCoordinatorServer()
	pb.RegisterGPUCoordinatorServer(grpcServer, coordServer)

	log.Println("GPU Coordinator server listening on port 50051")
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
