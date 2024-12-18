package main

import (
	"log"
	"net"

	"example.com/dsml/gpu_device_service"
	pb "example.com/dsml/proto"
	"google.golang.org/grpc"
)

func main() {
	ports := []string{"5003", "5004", "5005"} // GPU device servers on ports 5000-5002
	for i, port := range ports {
		deviceId := uint64(i + 1) // Assign unique device IDs (1, 2, 3)
		go func(p string, id uint64) {
			lis, err := net.Listen("tcp", "localhost:"+p)
			if err != nil {
				log.Fatalf("failed to listen on port %s: %v", p, err)
			}

			server := grpc.NewServer()
			pb.RegisterGPUDeviceServer(server, gpu_device_service.NewGPUDeviceServer(id, 0x3000)) // Unique deviceId
			log.Printf("GPU Device server listening on port %s with device ID %d", p, id)

			if err := server.Serve(lis); err != nil {
				log.Fatalf("failed to serve: %v", err)
			}
		}(port, deviceId)
	}
	select {} // Keep the main process running
}
