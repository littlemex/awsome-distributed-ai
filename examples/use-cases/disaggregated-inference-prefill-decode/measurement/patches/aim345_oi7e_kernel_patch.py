"""Disposable control for OPEN-ITEMS-NODE7 Q2: in SGLang's triton extend-attention _fwd_kernel, a program whose
query block starts at or beyond its own sequence's extend length returns immediately. The launch grid is
(batch, heads, cdiv(max_len_extend, BLOCK_M)) for every sequence, so without this a short sequence (for example a
64-token chunk remainder with an 8,128-token prefix) runs the full prefix loop in every one of the max-length
blocks with all rows masked. The early return changes no stored value (those programs store nothing valid)."""
import pathlib
import sglang
p = pathlib.Path(sglang.__file__).parent / "srt" / "layers" / "attention" / "triton_ops" / "extend_attention.py"
s = p.read_text()
old = """    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend
"""
assert s.count(old) == 1
s = s.replace(old, old + """    if cur_block_m * BLOCK_M >= cur_seq_len_extend:
        return
""", 1)
p.write_text(s)
print("oi7e kernel early-exit patch applied")
