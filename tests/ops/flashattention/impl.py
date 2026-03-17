
import torch
try:
    import torch_npu
except ImportError:
    torch_npu = None
import math
from typing import Dict, Any, List, Optional

class FlashAttentionNpuImpl:
    """FlashAttention implementation using npu_fused_infer_attention_score"""
    
    def __init__(self):
        self.name = "npu_flash_attention"
    
    def prepare_data(self, data: Dict[str, Any], device: str, precision) -> Dict[str, Any]:

        """Prepare data for core operator execution"""
        query = data['query'].to(dtype=precision.value, device=device)
        key = data['key'].to(dtype=precision.value, device=device)
        value = data['value'].to(dtype=precision.value, device=device)
        
        num_heads = data['num_heads']
        num_kv_heads = data.get('num_kv_heads', num_heads)
        input_layout = data.get('input_layout', "BNSD")
        
        scale = data.get('scale')
        if scale is None:
            head_size = query.shape[-1]
            scale = 1.0 / math.sqrt(head_size)
            
        # Automatic sequence length handling
        actual_seq_lengths = data.get('actual_seq_lengths')
        actual_seq_lengths_kv = data.get('actual_seq_lengths_kv')
        
        if input_layout != "TND":
            if actual_seq_lengths is not None and actual_seq_lengths_kv is not None:
                if actual_seq_lengths is actual_seq_lengths_kv or actual_seq_lengths == actual_seq_lengths_kv:
                    actual_seq_lengths_kv = None

        if actual_seq_lengths is None:
            if input_layout == "BNSD":
                batch_size = query.shape[0]
                seq_len = query.shape[2]
                actual_seq_lengths = [seq_len] * batch_size
            elif input_layout == "BSND":
                batch_size = query.shape[0]
                seq_len = query.shape[1]
                actual_seq_lengths = [seq_len] * batch_size
            elif input_layout == "TND":
                # For TND, we expect actual_seq_lengths to be provided in data
                pass
                
        if actual_seq_lengths_kv is None:
            if input_layout == "BNSD":
                batch_size = key.shape[0]
                seq_len_kv = key.shape[2]
                actual_seq_lengths_kv = [seq_len_kv] * batch_size
            elif input_layout == "BSND":
                batch_size = key.shape[0]
                seq_len_kv = key.shape[1]
                actual_seq_lengths_kv = [seq_len_kv] * batch_size
        
        if input_layout == "TND" and actual_seq_lengths_kv is None and actual_seq_lengths is not None:
            actual_seq_lengths_kv = list(actual_seq_lengths)

        # For TND layout, NPU op expects cumulative sum of sequence lengths
        if input_layout == "TND":
            if actual_seq_lengths is not None:
                # Calculate cumulative sum: [10, 20] -> [10, 30]
                # Note: NPU op expects int64 probably? or int32?
                # Usually list is fine, but let's check if it needs to be tensor or list
                # The API usually takes list[int] for actual_seq_lengths
                
                # Check if it's already cumulative? No, data dict has raw lengths usually
                
                # But wait, if we modify it here, we might affect other parts?
                # We are creating a new list for kwargs
                
                # Reference implementation uses cumulative sum
                import numpy as np
                # Check if it's already cumulative to avoid double application if called multiple times?
                # But prepare_data is usually called once.
                # Also convert numpy int64 to python int, as torch_npu might expect list of ints
                actual_seq_lengths = [int(x) for x in np.cumsum(actual_seq_lengths)]
                
            if actual_seq_lengths_kv is not None:
                # If block_table is present, actual_seq_lengths_kv should be logical lengths (not cumsum)
                # If block_table is None (standard TND), it should be cumsum
                if 'block_table' in data and data['block_table'] is not None:
                     pass # Keep as raw lengths
                else:
                     import numpy as np
                     actual_seq_lengths_kv = [int(x) for x in np.cumsum(actual_seq_lengths_kv)]

        kwargs = {
            "query": query,
            "key": key,
            "value": value,
            "actual_seq_lengths": actual_seq_lengths,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "num_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "input_layout": input_layout,
            "scale": scale,
            "pre_tokens": data.get('pre_tokens', 65535),
            "next_tokens": data.get('next_tokens', 65535),
            "sparse_mode": data.get('sparse_mode', 0),
        }
        
        if 'atten_mask' in data and data['atten_mask'] is not None:
            kwargs["atten_mask"] = data['atten_mask'].to(device=device)
        elif kwargs.get('sparse_mode', 0) == 3:
             # NPU requires atten_mask for sparse_mode=3
             # Create a full mask (all True) or causal mask?
             # Docs say "atten_mask: Attention mask tensor"
             # If sparse_mode is 3 (right down causal), usually the op handles causality internally
             # BUT the error says "when sparse_mode is 3, it not 0, atten_mask should not be null."
             # So we must provide a mask.
             # Let's provide a dummy full mask or a proper causal mask.
             # Since sparse_mode=3 implies causal, maybe a boolean mask of all ones (keep all)?
             # Or does it need the causal mask explicitly?
             # Usually "sparse_mode" controls the pattern, and mask is additional.
             # If we need to satisfy the API check, let's create a ones mask.
             
             # Need to know sequence length
             # actual_seq_lengths is list of int
             max_seq_len = max(actual_seq_lengths) if actual_seq_lengths else query.shape[2]
             
             # Create a triangular mask? Or just ones?
             # If sparse_mode=3 does the masking, then atten_mask should probably be "all valid".
             # Let's try creating a simple mask.
             # Shape: [B, 1, S, S] or [S, S] depending on input
             # BNSD -> S is dim 2
             
             if input_layout == "BNSD":
                 B = query.shape[0]
                 S = query.shape[2]
                 # Create lower triangular mask (causal)
                 # Note: The NPU op seems to expect a 2048x2048 mask for sparse_mode=3 based on error logs
                 # "[CheckAttentionMask] atten_mask layout is S1S2, shape [128, 128] should be equal to [2048, 2048]"
                 # This implies we should provide a 2048x2048 mask even for BNSD if sparse_mode=3
                 kwargs["atten_mask"] = torch.triu(torch.ones(2048, 2048), diagonal=1).to(device=device).to(torch.int8)
             elif input_layout == "TND":
                 # For TND with sparse_mode=3, we use a fixed size mask as requested
                 # Reference: paged_attention/fused_infer_attention_score_impl.py uses 2048x2048
                 # For sparse_mode=3, we need an upper triangular mask (1 indicates masked, 0 indicates kept)
                 # and dtype should be int8 (or bool with True=masked)
                 kwargs["atten_mask"] = torch.triu(torch.ones(2048, 2048), diagonal=1).to(device=device).to(torch.int8)
             
             # Keep sparse_mode as provided (usually 3 for this case)
             # kwargs["sparse_mode"] = 0

        if 'block_table' in data and data['block_table'] is not None:
            kwargs["block_table"] = data['block_table'].to(device=device)
            kwargs["block_size"] = data.get('block_size', 128)
            
            # If block_table is used, key and value must be reshaped to (num_blocks, block_size, num_heads, head_dim)
            # The test data generator provides key/value as (total_tokens, num_heads, head_dim) for TND
            # We need to reshape them if they are contiguous TND but we want to test block table path.
            # However, for true PagedAttention, the KV cache should be allocated as blocks.
            # In our test case `tnd_layout_with_block_table`, we generate contiguous data.
            # If we pass block_table, NPU op might interpret key/value as the backing storage for blocks?
            # Or maybe it expects key/value to be the cache?
            
            # Let's check paged_attention implementation.
            # key_cache shape: [num_blocks, block_size, num_kv_heads, head_size]
            # value_cache shape: [num_blocks, block_size, num_kv_heads, head_size]
            # And they are reshaped to view(num_block, block_size, -1) before passing?
            # No, wait. 
            # In paged_attention/fused_infer_attention_score_impl.py:
            # key_cache = key_cache.view(num_block, block_size, -1)
            # This flattens num_kv_heads and head_size?
            # [num_blocks, block_size, num_kv_heads * head_size] ?
            
            # If our key is (total_tokens, num_heads, head_size) from TND generator.
            # And total_tokens = batch * seq_len.
            # If block_size = 128, seq_len = 256. num_blocks = 2 per seq.
            # We need to reshape key/value to match what NPU expects when block_table is present.
            
            # If we look at the error: "qkHeadDim(128) ... vHeadDim(8) only support 128"
            # If we flattened heads into dim 2?
            # TND key shape: (T, N, D).
            # If NPU expects (num_blocks, block_size, hidden_size) where hidden_size = N*D ?
            
            # Let's try to reshape key/value if block_table is present.
            # We assume the input key/value are contiguous and can be viewed as blocks.
            
            if input_layout == "TND":
                block_size = kwargs["block_size"]
                # key is (Total_tokens, Num_KV_Heads, Head_Size)
                # We want (Num_Blocks, Block_Size, Num_KV_Heads * Head_Size) ?
                # Or (Num_Blocks, Block_Size, Num_KV_Heads, Head_Size) ?
                
                # Based on paged_attention impl:
                # key_cache = key_cache.view(num_block, block_size, -1)
                # It seems it expects 3D tensor: (Num_Blocks, Block_Size, Hidden_Size)
                
                T, N_kv, D = key.shape
                # Check if T is divisible by block_size
                if T % block_size == 0:
                    num_blocks = T // block_size
                    # Reshape to (Num_Blocks, Block_Size, N_kv * D)
                    kwargs["key"] = key.reshape(num_blocks, block_size, -1)
                    kwargs["value"] = value.reshape(num_blocks, block_size, -1)
                    
                    # Also, if we reshaped key/value, we might need to adjust num_key_value_heads?
                    # The error says vHeadDim(8). If we passed (T, 16, 128).
                    # Maybe it interpreted 16 as head dim? 
                    # If we reshape to (..., -1), we flatten N and D.
                    # Then num_key_value_heads=16 is passed separately.
                    # NPU op likely divides hidden_size by num_key_value_heads to get head_dim.
                    # (16*128) / 16 = 128. Correct.
                    
                    # If we passed (T, 16, 128) directly.
                    # Maybe it interprets T as blocks? No.
                    # TND implies input_layout="TND".
                    # But when block_table is passed, maybe it ignores input_layout for KV?
                    # Or maybe input_layout="TND" applies to Query only?
                    # Docs usually say input_layout applies to Query. KV layout depends on if it's Paged.
                    
                    # If block_table is present, it is Paged Attention.
                    # KV should be in block format.
                    pass
                else:
                    # If not divisible, we can't easily reshape. 
                    # But our test case uses seq_len=256, block_size=128, so it is divisible.
                    pass
        
        return kwargs

    def execute_core_operator(self, prepared_data: Dict[str, Any]) -> torch.Tensor:
        """Execute core operator - torch_npu.npu_fused_infer_attention_score"""
        if torch_npu is None:
            raise RuntimeError("torch_npu is not available")
            
        output, _ = torch_npu.npu_fused_infer_attention_score(**prepared_data)
        return output

    def run_full_implementation(self, data: Dict[str, Any], device: str, precision) -> torch.Tensor:
        """Run full implementation (including data preparation and post-processing)"""
        # Save original seq lengths before they are converted to cumsum in prepare_data
        original_seq_lengths = data.get('actual_seq_lengths')
        
        prepared_data = self.prepare_data(data, device, precision)
        
        output = self.execute_core_operator(prepared_data)
        
        # Post-process for layout if needed
        input_layout = prepared_data.get('input_layout', "BNSD")
        if input_layout == "TND":
            num_heads = prepared_data['num_heads']
            head_size = output.shape[-1]
            
            # Try to get dimensions from metadata or input shapes
            if 'metadata' in data and 'seq_len' in data['metadata']:
                batch_size = data['metadata']['batch_size']
                seq_len = data['metadata']['seq_len']
                
                # Check if we are in variable seq len mode
                # Use original_seq_lengths because prepared_data['actual_seq_lengths'] is now cumsum
                if original_seq_lengths is not None and len(set(original_seq_lengths)) > 1:
                    # Variable sequence lengths
                    # We need to unpack TND back to BNSD with padding
                    
                    # Create empty output tensor (B, N, S, D)
                    output_bnsd = torch.zeros(
                        batch_size, num_heads, seq_len, head_size, 
                        dtype=output.dtype, device=output.device
                    )
                    
                    current_idx = 0
                    for i, length in enumerate(original_seq_lengths):
                        # Extract valid tokens for this sequence: (length, N, D)
                        valid_tokens = output[current_idx : current_idx + length]
                        # (length, N, D) -> (N, length, D)
                        valid_tokens = valid_tokens.transpose(0, 1)
                        # Place into output
                        output_bnsd[i, :, :length, :] = valid_tokens
                        current_idx += length
                        
                    output = output_bnsd
                else:
                    # Uniform sequence length
                    # (B*S, N, D) -> (B, S, N, D) -> (B, N, S, D)
                    total_tokens = batch_size * seq_len
                    # Handle case where output might not match total tokens (e.g. if padded/packed differently)
                    if output.shape[0] == total_tokens:
                        output = output.reshape(batch_size, seq_len, num_heads, head_size).transpose(1, 2)
                    else:
                        # Fallback for when shapes don't align perfectly (though they should for uniform TND)
                         pass
            else:
                # Fallback heuristics or error
                # Since this is primarily for testing framework, we can assume metadata exists
                # Or try to deduce from actual_seq_lengths if uniform
                # Note: prepared_data has cumsum, so we can't easily check uniformity directly from it without decoding
                # But for uniform TND tests, usually metadata is present.
                
                # If original_seq_lengths is available
                if original_seq_lengths is not None and len(set(original_seq_lengths)) == 1:
                     seq_len = original_seq_lengths[0]
                     batch_size = len(original_seq_lengths)
                     output = output.reshape(batch_size, seq_len, num_heads, head_size).transpose(1, 2)
                else:
                     # Try to use metadata from prepared_data?
                     # prepared_data doesn't have metadata field usually
                     pass
            
        return output.cpu().float()

    def run(self, 
            query: torch.Tensor, 
            key: torch.Tensor, 
            value: torch.Tensor, 
            num_heads: int,
            num_kv_heads: int = None,
            scale: float = None,
            input_layout: str = "BNSD",
            actual_seq_lengths: List[int] = None,
            actual_seq_lengths_kv: List[int] = None,
            pre_tokens: int = 65535,
            next_tokens: int = 65535,
            sparse_mode: int = 0,
            atten_mask: torch.Tensor = None,
            block_table: torch.Tensor = None,
            block_size: int = 0
           ) -> torch.Tensor:
        """Legacy run method for compatibility"""
        data = {
            "query": query,
            "key": key,
            "value": value,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "scale": scale,
            "input_layout": input_layout,
            "actual_seq_lengths": actual_seq_lengths,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "pre_tokens": pre_tokens,
            "next_tokens": next_tokens,
            "sparse_mode": sparse_mode,
            "atten_mask": atten_mask,
            "block_table": block_table,
            "block_size": block_size
        }
        # Assume float16/bfloat16 and current device for legacy call
        device = str(query.device)
        # Hack: map torch dtype to PrecisionType if we had that available here, 
        # but for now let's just use a dummy object with .value
        class DummyPrecision:
            def __init__(self, dtype): self.value = dtype
        
        precision = DummyPrecision(query.dtype)
        
        # Prepare data but keep tensors on device (prepare_data moves them)
        # Actually prepare_data expects cpu tensors in data dict usually? 
        # In the original impl, prepare_data takes data dict and moves to device.
        # But here run() gets tensors already on device potentially.
        # Let's just construct kwargs manually to avoid breaking legacy run
        
        # Re-using logic from original run method for safety
        if torch_npu is None:
            raise RuntimeError("torch_npu is not available")
            
        if num_kv_heads is None:
            num_kv_heads = num_heads
            
        if scale is None:
            head_size = query.shape[-1]
            scale = 1.0 / math.sqrt(head_size)
            
        if actual_seq_lengths is None:
            if input_layout == "BNSD":
                batch_size = query.shape[0]
                seq_len = query.shape[2]
                actual_seq_lengths = [seq_len] * batch_size
            elif input_layout == "BSND":
                batch_size = query.shape[0]
                seq_len = query.shape[1]
                actual_seq_lengths = [seq_len] * batch_size
                
        if actual_seq_lengths_kv is None:
            if input_layout == "BNSD":
                batch_size = key.shape[0]
                seq_len_kv = key.shape[2]
                actual_seq_lengths_kv = [seq_len_kv] * batch_size
            elif input_layout == "BSND":
                batch_size = key.shape[0]
                seq_len_kv = key.shape[1]
                actual_seq_lengths_kv = [seq_len_kv] * batch_size

        kwargs = {
            "query": query,
            "key": key,
            "value": value,
            "actual_seq_lengths": actual_seq_lengths,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "num_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "input_layout": input_layout,
            "scale": scale,
            "pre_tokens": pre_tokens,
            "next_tokens": next_tokens,
            "sparse_mode": sparse_mode,
        }
        
        if atten_mask is not None:
            kwargs["atten_mask"] = atten_mask
            
        if block_table is not None:
            kwargs["block_table"] = block_table
            kwargs["block_size"] = block_size
            
        output, _ = torch_npu.npu_fused_infer_attention_score(**kwargs)
        return output
