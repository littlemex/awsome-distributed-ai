#!/usr/bin/env python3
"""Fixed-total-bytes transfer split into N descriptors, one single initialize_xfer/transfer()
call per N (mirrors how SGLang's nixl/conn.py posts one transfer with all of a request's
descriptors). Server and client run as two separate OS processes on the same node (loopback TCP
for metadata exchange only -- the data path is NIXL/libfabric or NIXL/UCX, same as fabric_probe.py).
Not an SGLang connector; a diagnostic microbenchmark derived from fabric_probe.py."""
import argparse
import base64
import json
import socket
import time
import uuid


def send(file, data):
    file.write(json.dumps(data).encode() + b'\n')
    file.flush()


def receive(file):
    line = file.readline(64 * 1024 * 1024)
    if not line:
        raise RuntimeError('Peer disconnected')
    return json.loads(line)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('role', choices=['server', 'client'])
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, default=19100)
    p.add_argument('--total-bytes', type=int, default=12_800_000)
    p.add_argument('--descriptor-counts', default='1,100,1000,10000,100000')
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--backend', choices=['LIBFABRIC', 'UCX'], default='LIBFABRIC')
    p.add_argument('--instance-type', required=True)
    a = p.parse_args()
    counts = [int(x) for x in a.descriptor_counts.split(',')]
    for n in counts:
        assert a.total_bytes % n == 0, f'total-bytes must be divisible by every descriptor count ({n})'
    import torch
    from nixl._api import nixl_agent, nixl_agent_config
    name = 'kvx-' + uuid.uuid4().hex
    agent = nixl_agent(name, nixl_agent_config(backends=[a.backend]))
    tensor = torch.full((a.total_bytes,), 17 if a.role == 'client' else 0, dtype=torch.uint8, device='cuda')
    torch.cuda.synchronize()
    registration = agent.register_memory(tensor)
    base = tensor.data_ptr()
    info = {'agent': name, 'metadata': base64.b64encode(agent.get_agent_metadata()).decode(),
            'address': base, 'bytes': a.total_bytes, 'gpu_index': tensor.get_device(), 'backend': a.backend}
    if a.role == 'server':
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((a.host, a.port))
        listener.listen(1)
        listener.settimeout(180)
        conn, _ = listener.accept()
    else:
        conn = socket.create_connection((a.host, a.port), timeout=180)
    conn.settimeout(180)
    results = []
    with conn, conn.makefile('rwb') as file:
        send(file, info)
        peer = receive(file)
        assert peer['bytes'] == a.total_bytes and peer['backend'] == a.backend
        agent.add_remote_agent(base64.b64decode(peer['metadata']))
        send(file, {'ready': True})
        assert receive(file)['ready']
        if a.role == 'client':
            for n in counts:
                chunk = a.total_bytes // n
                local = agent.get_xfer_descs([(base + i * chunk, chunk, tensor.get_device()) for i in range(n)], 'VRAM')
                remote = agent.get_xfer_descs([(peer['address'] + i * chunk, chunk, peer['gpu_index']) for i in range(n)], 'VRAM')
                handle = agent.initialize_xfer('WRITE', local, remote, peer['agent'], backends=[a.backend])
                elapsed = []
                for _ in range(a.iterations):
                    start = time.perf_counter()
                    agent.transfer(handle)
                    while True:
                        state = agent.check_xfer_state(handle)
                        if state == 'DONE':
                            break
                        if state != 'PROC' or time.perf_counter() - start > 120:
                            raise RuntimeError(f'n={n} transfer state {state}')
                        time.sleep(.0001)
                    elapsed.append(time.perf_counter() - start)
                agent.release_xfer_handle(handle)
                row = {'instance_type': a.instance_type, 'evidence_scope': 'mechanism-validation',
                       'backend': a.backend, 'descriptor_count': n, 'bytes_per_descriptor': chunk,
                       'total_bytes': a.total_bytes, 'iterations_count': a.iterations,
                       'first_transfer_ms': elapsed[0] * 1000,
                       'warm_transfer_mean_ms': sum(elapsed[1:]) / len(elapsed[1:]) * 1000,
                       'warm_transfer_ms_per_descriptor': sum(elapsed[1:]) / len(elapsed[1:]) * 1000 / n,
                       'warm_gib_per_s': a.total_bytes * len(elapsed[1:]) / sum(elapsed[1:]) / 1024 ** 3}
                print(json.dumps(row), flush=True)
                results.append(row)
            send(file, {'transferred': True, 'counts': counts})
            verification = receive(file)
            assert verification['correct'], verification
        else:
            req = receive(file)
            torch.cuda.synchronize()
            correct = bool(torch.all(tensor == 17).item())
            send(file, {'correct': correct})
            assert correct
    agent.deregister_memory(registration)


if __name__ == '__main__':
    main()
