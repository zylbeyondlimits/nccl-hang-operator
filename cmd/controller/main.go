package main

import (
	"flag"
	"log"
	"time"
)

func main() {
	grpcAddr := flag.String("telemetry-addr", ":9090", "gRPC listen address for node-agent rank heartbeats")
	metricsAddr := flag.String("metrics-addr", ":8080", "Prometheus metrics address")
	reconcileTick := flag.Duration("reconcile-interval", 10*time.Second, "Base reconcile polling interval")
	flag.Parse()

	log.Printf("nccl-hang-operator controller initialized telemetry=%s metrics=%s interval=%s",
		*grpcAddr, *metricsAddr, *reconcileTick)
}
