"""Time one KV-sized NIXL WRITE between two GPUs of one node, outside SGLang.

Each pair runs in its own process (sender agent on GPU src, receiver agent on GPU dst), and the
pairs start together, like the two prefill TP ranks of one request. Reports, per iteration, the
time initialize_xfer+transfer took to return (post) and the time from that return until
check_xfer_state first reads DONE (wait), polled in a tight loop with no other Python thread.
"""
import argparse
import json
import multiprocessing as mp
import time


def pair(idx, src_gpu, dst_gpu, layers, tokens, bytes_per_token_layer, pieces, iters, backend, num_threads, stagger_ms, iface, barrier, out):
    import os
    if iface:
        os.environ["FI_EFA_IFACE"] = iface
    import numpy as np
    import torch
    from nixl._api import nixl_agent, nixl_agent_config

    def agent(name):
        a = nixl_agent(name, nixl_agent_config(backends=[], num_threads=num_threads))
        a.create_backend(backend, {})
        return a

    layer_bytes = tokens * bytes_per_token_layer
    src = torch.empty(layers * layer_bytes, dtype=torch.uint8, device=f"cuda:{src_gpu}")
    dst = torch.empty(layers * layer_bytes, dtype=torch.uint8, device=f"cuda:{dst_gpu}")
    src.fill_(7)
    torch.cuda.synchronize(src_gpu)
    snd, rcv = agent(f"snd{src_gpu}"), agent(f"rcv{dst_gpu}")
    snd.register_memory([(src.data_ptr(), src.numel(), src_gpu, "")], "VRAM")
    rcv.register_memory([(dst.data_ptr(), dst.numel(), dst_gpu, "")], "VRAM")
    peer = snd.add_remote_agent(rcv.get_agent_metadata())
    piece = layer_bytes // pieces
    offs = np.array([l * layer_bytes + p * piece for l in range(layers) for p in range(pieces)], dtype=np.uint64)
    lens = np.full_like(offs, piece)

    def reqs(base, gpu):
        return np.column_stack((offs + np.uint64(base), lens, np.full_like(offs, gpu)))

    sd = snd.get_xfer_descs(reqs(src.data_ptr(), src_gpu), "VRAM")
    dd = snd.get_xfer_descs(reqs(dst.data_ptr(), dst_gpu), "VRAM")
    rows = []
    for i in range(iters + 3):
        barrier.wait()
        if stagger_ms and idx:
            time.sleep(stagger_ms * idx / 1000.0)
        t0 = time.perf_counter()
        h = snd.initialize_xfer("WRITE", sd, dd, peer, f"bench{i}".encode())
        assert snd.transfer(h) != "ERR"
        t1 = time.perf_counter()
        polls = 0
        while True:
            s = snd.check_xfer_state(h)
            polls += 1
            if s == "DONE":
                break
            assert s != "ERR"
        t2 = time.perf_counter()
        snd.release_xfer_handle(h)
        if i >= 3:
            rows.append({"post_ms": (t1 - t0) * 1e3, "wait_ms": (t2 - t1) * 1e3, "polls": polls, "t0": t0, "t2": t2})
    torch.cuda.synchronize(dst_gpu)
    assert int(dst[0].item()) == 7 and int(dst[-1].item()) == 7
    out.put({"src": src_gpu, "dst": dst_gpu, "bytes": int(offs.size * piece), "descriptors": int(offs.size), "rows": rows})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="0:2,1:3")
    p.add_argument("--layers", type=int, default=27)
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--bytes-per-token-layer", type=int, default=1152)
    p.add_argument("--pieces", type=int, default=1)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--backend", default="LIBFABRIC")
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--stagger-ms", type=float, default=0.0)
    p.add_argument("--ifaces", default="", help="comma-separated FI_EFA_IFACE per pair; empty keeps the default")
    a = p.parse_args()
    mp.set_start_method("spawn")
    pairs = [tuple(int(x) for x in s.split(":")) for s in a.pairs.split(",")]
    barrier, out = mp.Barrier(len(pairs)), mp.Queue()
    procs = [mp.Process(target=pair, args=(i, s, d, a.layers, a.tokens, a.bytes_per_token_layer, a.pieces, a.iters, a.backend, a.num_threads, a.stagger_ms, (a.ifaces.split(",") + [""] * len(pairs))[i] if a.ifaces else "", barrier, out)) for i, (s, d) in enumerate(pairs)]
    for x in procs:
        x.start()
    res = [out.get(timeout=900) for _ in procs]
    for x in procs:
        x.join()
    print(json.dumps({"args": vars(a), "pairs": res}))


if __name__ == "__main__":
    main()
