#!/usr/bin/env bash
set -euo pipefail
# Only the dedicated demo ports/namespace are changed. USB retains management access.
mode=${BEV_LINK_MODE:-software}
host_port=${BEV_LINK_IFACE:-bev-host}
edge_port=${BEV_EDGE_IFACE:-bev-edge0}
host_cidr=${BEV_HOST_CIDR:-10.78.0.1/30}
edge_cidr=${BEV_EDGE_CIDR:-10.78.0.2/30}
original_mtu=${BEV_ORIGINAL_MTU:-1500}
for port in "$host_port" "$edge_port"; do
    [[ $port =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid demo interface'; exit 1; }
done
if [[ ${1:-start} == stop ]]; then
    if [[ $mode == physical ]]; then
        python3 /home/seeed/jetson-bev-demo/jetson/tune_network.py stop
        if ip netns exec bev-edge ip link show "$edge_port" >/dev/null 2>&1; then
            ip netns exec bev-edge ip link set "$edge_port" down
            ip netns exec bev-edge ip addr del "$edge_cidr" dev "$edge_port" 2>/dev/null || true
            ip netns exec bev-edge ip link set "$edge_port" netns 1
        fi
        ip addr del "$host_cidr" dev "$host_port" 2>/dev/null || true
        for port in "$host_port" "$edge_port"; do
            ip link set "$port" down
            ip link set "$port" mtu "$original_mtu"
            nmcli device set "$port" managed yes
        done
    fi
    ip netns del bev-edge 2>/dev/null || true
    ip link del bev-host 2>/dev/null || true
    exit 0
fi
# Keep the two reported crash-triggering ports down, including after reboot.
# This is runtime interface isolation, not a BSP/PHY repair.
if [[ $mode != physical && -n ${BEV_DISABLED_PORTS:-} ]]; then
    IFS=, read -ra disabled_ports <<< "$BEV_DISABLED_PORTS"
    for port in "${disabled_ports[@]}"; do
        [[ $port =~ ^mgbe[0-3]_0$ ]] || { echo 'Invalid disabled physical port'; exit 1; }
        nmcli device set "$port" managed no
        ip link set "$port" down
    done
fi
ip netns list | awk '{print $1}' | grep -qx bev-edge || ip netns add bev-edge
ip netns exec bev-edge ip link set lo up
if [[ $mode == physical ]]; then
    [[ $host_port != "$edge_port" ]] || { echo 'Two different physical ports are required'; exit 1; }
    nmcli device set "$host_port" managed no
    if ip link show "$edge_port" >/dev/null 2>&1; then
        nmcli device set "$edge_port" managed no
        ip link set "$edge_port" netns bev-edge
    fi
    # The NVIDIA driver requires ports down when changing MTU.
    ip link set "$host_port" down
    ip netns exec bev-edge ip link set "$edge_port" down
    ip link set "$host_port" mtu "${BEV_LINK_MTU:-9000}"
    ip netns exec bev-edge ip link set "$edge_port" mtu "${BEV_LINK_MTU:-9000}"
    ip addr replace "$host_cidr" dev "$host_port"
    ip netns exec bev-edge ip addr replace "$edge_cidr" dev "$edge_port"
    ip link set "$host_port" up
    ip netns exec bev-edge ip link set "$edge_port" up
    # Disable the previous software route so all worker traffic must use the cable.
    if ip link show bev-host >/dev/null 2>&1; then
        ip addr del 10.77.0.1/30 dev bev-host 2>/dev/null || true
        ip link set bev-host down
        ip netns exec bev-edge ip addr del 10.77.0.2/30 dev bev-edge0 2>/dev/null || true
        ip netns exec bev-edge ip link set bev-edge0 down
    fi
    python3 /home/seeed/jetson-bev-demo/jetson/tune_network.py start
    echo "Physical Ethernet ready: $host_port ($host_cidr) <-> $edge_port ($edge_cidr)"
    exit 0
fi
if ! ip link show bev-host >/dev/null 2>&1; then
    ip link add bev-host type veth peer name bev-edge0
    ip link set bev-edge0 netns bev-edge
fi
ip addr replace 10.77.0.1/30 dev bev-host
ip link set bev-host up
ip netns exec bev-edge ip addr replace 10.77.0.2/30 dev bev-edge0
ip netns exec bev-edge ip link set bev-edge0 up
tc qdisc replace dev bev-host root tbf rate 1gbit burst 1mb latency 50ms
ip netns exec bev-edge tc qdisc replace dev bev-edge0 root tbf rate 1gbit burst 1mb latency 50ms
echo 'Software Ethernet ready: 10.77.0.1 <-> 10.77.0.2, 1 Gbps per direction'
