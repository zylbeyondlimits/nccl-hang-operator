package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/zylbeyondlimits/nccl-hang-operator/internal/telemetry"
)

func main() {
	nodeName := flag.String("node-name", os.Getenv("NODE_NAME"), "Kubernetes node name")
	btfPath := flag.String("btf-path", "/sys/kernel/btf/vmlinux", "Path to kernel BTF for CO-RE eBPF preflight")
	ibRoot := flag.String("ib-root", "/sys/class/infiniband", "Sysfs root for InfiniBand/RoCE counters")
	interval := flag.Duration("poll-interval", 10*time.Second, "Telemetry polling interval")
	flag.Parse()

	mode := telemetry.DetectCollectorMode(*btfPath)
	log.Printf("nccl-node-agent starting node=%q mode=%s poll=%s", *nodeName, mode, *interval)

	errs, err := telemetry.ReadIBPortErrors(*ibRoot)
	if err != nil {
		log.Printf("infiniband counters unavailable (%v); continuing with NVML/FlightRecorder signals", err)
	} else {
		fmt.Printf("initial_ib_port_errors=%d\n", errs)
	}
}
