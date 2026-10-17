#!/usr/bin/env python3
"""Fire N open-loop requests at a fixed rate against the router's /generate, to produce
'idle' vs 'loaded' conditions for reading SGLang's own (patched) KVX_DESC_COUNT log line. Runs
inside the engine pod (reaches the router Service by cluster DNS). Not a benchmark harness --
no percentiles, no traffic shape -- just enough load to compare descriptor counts at two rates."""
import argparse
import asyncio
import time
import httpx


async def one_call(client, url, input_ids, out_tokens, idx):
    body = {"input_ids": input_ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": out_tokens, "ignore_eos": True}, "stream": False}
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=body, timeout=120)
        r.raise_for_status()
        print(f"call {idx} ok in {time.perf_counter() - t0:.2f}s")
    except Exception as e:
        print(f"call {idx} FAILED after {time.perf_counter() - t0:.2f}s: {e}")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--rate", type=float, required=True, help="tasks/s, open-loop arrival")
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--input-tokens", type=int, default=8192)
    p.add_argument("--output-tokens", type=int, default=8)
    p.add_argument("--token-id", type=int, default=100)
    a = p.parse_args()
    interval = 1.0 / a.rate
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(a.count):
            # Vary every token by call index so no two calls share any prefix at all (defeats
            # prefix-cache dedup across the *whole* prompt, not just a tail -- the real traffic's
            # shape-b.json caps prefix_overlap_max_fraction at 0.05, i.e. mostly-fresh prompts).
            input_ids = [a.token_id + (i % 97) + (j % 53) for j in range(a.input_tokens)]
            tasks.append(asyncio.create_task(one_call(client, a.url, input_ids, a.output_tokens, i)))
            if i < a.count - 1:
                await asyncio.sleep(interval)
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
