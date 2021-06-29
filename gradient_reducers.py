# Heavily borrowed(almost copied) from code for PowerSGD.:
import datetime
import random
import os
import time
from contextlib import contextmanager
from typing import List
import numpy as np
import torch

try:
    import bit2byte
except ImportError:
    pass

num_times = 4
class Reducer:
    def __init__(self, random_seed, device, timer):
        self.rng = np.random.RandomState(random_seed)
        # M = 1024 * 1024
        # self.precalc_numbers = (
            # torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        # )
        if torch.distributed.is_available():
            self.n_workers = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0
        self.device = device
        self.timer = timer

    def reduce(self, grad_in, grad_out, memory_out):
        """Return communicated bits"""
        raise NotImplementedError()


class SignAndNormReducer(Reducer):
    """
    Optimizations:
    pack all weights in one big vector
    turn that to bits
    """
    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        sign_compressor = SignCompressor()

        with self.timer("reduce.flatpack"):
            flatgrad = TensorBuffer(grad_in)

        # Compute norms
        with self.timer("reduce.norms", verbosity=2):
            my_norms = torch.empty(len(grad_in), device=self.device)
            for i, tensor in enumerate(grad_in):
                my_norms[i] = tensor.norm(p=1)

        with self.timer("reduce.compress", verbosity=2):
            my_bits, sign_size = sign_compressor.compress(flatgrad.buffer)

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                bits = [torch.empty_like(my_bits) for i in range(self.n_workers)]
                norms = [torch.empty_like(my_norms) for i in range(self.n_workers)]
                h1 = all_gather(bits, my_bits, async_op=True)
                h2 = all_gather(norms, my_norms, async_op=True)
                h1.wait()
                h2.wait()
            else:
                bits = [my_bits]
                norms = [my_norms]

        bits_communicated += n_bits(my_bits)  # for the norm vector, being optimistic here
        bits_communicated += n_bits(my_norms)  # for the norm

        with self.timer("reduce.decompress", verbosity=2):
            flatsigns = []
            for their_bits in bits:
                uncompressed = sign_compressor.uncompress(their_bits, sign_size)
                flatsigns.append(uncompressed)

        with self.timer("reduce.average", verbosity=2):
            for out in grad_out:
                out.data[:] = 0.0

            for their_flatsigns, their_norms in zip(flatsigns, norms):
                flatgrad.buffer = their_flatsigns
                for sign, out, norm in zip(
                    flatgrad, grad_out, their_norms
                ):
                    out.data.add_(
                        norm / sign.nelement() / self.n_workers,
                        sign,
                    )

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, norm in zip(grad_in, memory_out, my_norms):
                mem.data[:] = tensor
                mem.data.add_(-norm / tensor.nelement(), tensor.sign())

        return bits_communicated


class SignReducer(Reducer):
    """
    Optimizations:
    pack all weights in one big vector
    turn that to bits
    """
    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        sign_compressor = SignCompressor()

        with self.timer("reduce.flatpack"):
            flatgrad = TensorBuffer(grad_in)

        with self.timer("reduce.compress", verbosity=2):
            my_bits, sign_size = sign_compressor.compress(flatgrad.buffer)

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                bits = [torch.empty_like(my_bits) for i in range(self.n_workers)]
                h1 = all_gather(bits, my_bits)
            else:
                bits = [my_bits]

        bits_communicated += n_bits(my_bits)  # for the norm vector, being optimistic here

        with self.timer("reduce.decompress", verbosity=2):
            flatsigns = []
            for their_bits in bits:
                uncompressed = sign_compressor.uncompress(their_bits, sign_size)
                flatsigns.append(uncompressed)

        with self.timer("reduce.average", verbosity=2):
            avg_flatsign = torch.stack(flatsigns).sum(dim=0) / self.n_workers
            flatgrad.buffer = avg_flatsign

            for out, avg in zip(grad_out, flatgrad):
                out.data[:] = avg

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem in zip(grad_in, memory_out):
                mem.data[:] = tensor
                mem.data.add_(tensor.sign())

        return bits_communicated


class SignSGDwithMajorityVoteReducer(Reducer):
    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        sign_compressor = SignCompressor()
        # start_time = torch.cuda.Event(enable_timing=True)
        # stop_time = torch.cuda.Event(enable_timing=True)
        # start_time.record()
        with self.timer("reduce.flatpack"):
            flatgrad = TensorBuffer(grad_in)
        # stop_time.record()
        # torch.cuda.synchronize()
        # print ("Time spent flatpack{}".format(
            # start_time.elapsed_time(stop_time)))
        # start_time.record()
        with self.timer("reduce.compress", verbosity=2):
            my_bits, sign_size = sign_compressor.compress(flatgrad.buffer)
        # stop_time.record()
        # torch.cuda.synchronize()
        # print ("Time spent compression{}".format(
            # start_time.elapsed_time(stop_time)))
        # start_time.record()
        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                bits = [torch.empty_like(my_bits) for i in range(self.n_workers)]
                all_gather(bits, my_bits)
            else:
                bits = [my_bits]
        # stop_time.record()
        # torch.cuda.synchronize()
        # print("Time all gather {}".format(start_time.elapsed_time(stop_time)))
        bits_communicated += n_bits(my_bits)  # for the norm vector, being optimistic here
        
        # start_time.record()
        with self.timer("reduce.decompress", verbosity=2):
            sum_of_signs = None
            for t in range(1):
                for their_bits in bits:
                    uncompressed = sign_compressor.uncompress(their_bits, sign_size)
                    if sum_of_signs is None:
                        sum_of_signs = uncompressed
                    else:
                        sum_of_signs += uncompressed
        # stop_time.record()
        # torch.cuda.synchronize()
        # print ("Time for decompress{}".format(
            # start_time.elapsed_time(stop_time)))
        # start_time.record()
        with self.timer("reduce.majorityvote", verbosity=2):
            total_sign = sum_of_signs.sign()
        # stop_time.record()
        # torch.cuda.synchronize()
        # print("Time for majority vote{}".format( start_time.elapsed_time(stop_time)))
        # start_time.record()
        with self.timer("reduce.set_out", verbosity=2):
            flatgrad.buffer = total_sign
            for out, majorityvote in zip(grad_out, flatgrad):
                out.data[:] = majorityvote
        # stop_time.record()
        # torch.cuda.synchronize()
        # print ("Time for copy data back{}".format(
            # start_time.elapsed_time(stop_time)))
        with self.timer("reduce.memory", verbosity=2):
            for mem in memory_out:
                mem.data[:] = -10_000_000  # don't try to use memory
        # print ("Bit communication{}\n".format( bits_communicated))
        return bits_communicated

class TopKReducer(Reducer):
    """
    Use same amount as rank-based
    """
    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.compression = 0.2

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        
        # start_time = torch.cuda.Event(enable_timing=True)
        # stop_time = torch.cuda.Event(enable_timing=True)

        # start_time.record()
        with self.timer("reduce.flatpack", verbosity=2):
            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]
            for tensor in grad_in:
                top_size = max(1, int(0.5 * self.compression * tensor.nelement()))
                flatgrad_size += top_size
                tensor_idx.append(tensor_idx[-1] + top_size)
            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            flat_values = torch.empty(flatgrad_size, device=self.device)
            flat_positions = torch.empty(flatgrad_size, device=self.device, dtype=torch.int)
        # end_time.record()
        # torch.cuda.synchronize()
        # print ("Time flatpack {}".format(start_time.elapsed_time(stop_time)))
        # start_time.record()
        with self.timer("reduce.topk", verbosity=2):
            for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):
                top_size = max(1, int(0.5 * self.compression * tensor.nelement()))
                _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)
                values = tensor.view(-1)[positions].contiguous()
                flat_values[start:end] = values
                flat_positions[start:end] = positions
        # stop_time.record()
        # torch.cuda.synchronize()
        # print ("Time topk sort {}".format(start_time.elapsed_time(stop_time)))

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, start, end in zip(
                grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                positions = flat_positions[start:end]
                mem.data[:] = tensor
                mem.view(-1)[positions.long()] = 0.0

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(flat_values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(flat_positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, flat_values, async_op=True)
                h2 = all_gather(worker_positions, flat_positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [flat_values]
                worker_positions = [flat_positions]
            bits_communicated += n_bits(flat_values) + n_bits(flat_positions)

        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, start, end in zip(
                grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                out.data[:] = 0
                for pos, val in zip(worker_positions, worker_values):
                    positions = pos[start:end]
                    values = val[start:end]
                    # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
                    out.view(-1)[positions.long()] += values / self.n_workers
        # print ("Bit bits_communicated {}".format(bits_communicated))
        return bits_communicated


class GlobalTopKReducer(Reducer):
    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.compression = compression

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        with self.timer("reduce.flatpack"):
            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]
            for tensor in grad_in:
                n = tensor.nelement()
                flatgrad_size += n
                tensor_idx.append(tensor_idx[-1] + n)
            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            flatgrad = torch.empty(flatgrad_size, device=self.device)

            # Pack the flatgrad
            for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):
                flatgrad[start:end] = tensor.view(-1)

        top_size = max(1, int(0.5 * self.compression * flatgrad.nelement()))

        with self.timer("reduce.topk", verbosity=2):
            _, positions = torch.topk(flatgrad.abs(), top_size, sorted=False)
            values = flatgrad[positions].contiguous()

        with self.timer("reduce.set_memory", verbosity=2):
            for tensor, mem, start, end in zip(
                grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                local_positions = positions[(positions >= start) & (positions < end)] - start
                mem.data[:] = tensor
                mem.view(-1)[local_positions] = 0.0

        with self.timer("reduce.reduce", verbosity=2):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, values, async_op=True)
                h2 = all_gather(worker_positions, positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [values]
                worker_positions = [positions]
            bits_communicated += n_bits(values) + n_bits(positions)

        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, start, end in zip(
                grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                out.data[:] = 0.0
                for pos, val in zip(worker_positions, worker_values):
                    local_positions = pos[(pos >= start) & (pos < end)] - start
                    local_vals = val[(pos >= start) & (pos < end)]
                    out.view(-1)[local_positions] += local_vals / self.n_workers

        return bits_communicated


class UniformRandomSparseBlockReducer(Reducer):
    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.compression = compression

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        values_list = []
        start_idx_list = []
        block_sizes = []

        with self.timer("reduce.block", verbosity=2):
            for tensor in grad_in:
                block_size = max(1, int(self.compression * tensor.nelement()))
                block_sizes.append(block_size)
                start_idx = self.rng.choice(tensor.nelement())
                start_idx_list.append(start_idx)
                end_idx = min(start_idx + block_size, tensor.nelement())
                bfr = torch.empty(block_size, dtype=torch.float32, device=self.device)
                bfr[: end_idx - start_idx] = tensor.view(-1)[start_idx:end_idx]
                rest = block_size - (end_idx - start_idx)
                if rest > 0:
                    bfr[end_idx - start_idx :] = tensor.view(-1)[:rest]
                values_list.append(bfr)

        with self.timer("reduce.flatpack", verbosity=2):
            flat_values = TensorBuffer(values_list)

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, start_idx, block_size in zip(grad_in, memory_out, start_idx_list, block_sizes):
                end_idx = min(start_idx + block_size, tensor.nelement())
                rest = block_size - (end_idx - start_idx)
                mem.data = tensor.clone()
                mem.view(-1)[start_idx:end_idx] = 0.0
                if rest > 0:
                    mem.view(-1)[:rest] = 0.0

        with self.timer("reduce.reduce", verbosity=2):
            flat_values.all_reduce()
            flat_values.buffer /= self.n_workers
            bits_communicated += flat_values.bits()

        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, start_idx, block_size, values in zip(grad_in, grad_out, start_idx_list, block_sizes, flat_values):
                end_idx = min(start_idx + block_size, tensor.nelement())
                rest = block_size - (end_idx - start_idx)
                out.data.zero_()
                out.view(-1)[start_idx:end_idx] = values[: end_idx - start_idx]
                if rest > 0:
                    out.view(-1)[:rest] = values[end_idx - start_idx :]

        return bits_communicated


class UniformRandomSparseReducer(Reducer):
    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.compression = compression

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        indices_list = []
        values_list = []

        with self.timer("reduce.block", verbosity=2):
            for tensor in grad_in:
                block_size = max(1, int(self.compression * tensor.nelement()))
                indices = self.rng.choice(tensor.nelement(), block_size, replace=False)
                indices_list.append(indices)
                values = tensor.view(-1)[indices]
                values_list.append(values)

        with self.timer("reduce.flatpack", verbosity=2):
            flat_values = TensorBuffer(values_list)

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, indices in zip(grad_in, memory_out, indices_list):
                mem.data[:] = tensor
                mem.view(-1)[indices] = 0.0

        with self.timer("reduce.reduce", verbosity=2):
            flat_values.all_reduce()
            flat_values.buffer.data /= self.n_workers
            bits_communicated += flat_values.bits()

        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, values, indices in zip(grad_in, grad_out, flat_values, indices_list):
                out.data.zero_()
                out.view(-1)[indices] = values

        return bits_communicated


class RandomSparseBlockReducer(Reducer):
    def __init__(self, random_seed, device, timer, rank):
        super().__init__(random_seed, device, timer)
        self.rank = rank

    def block_size(self, tensor):
        # return max(1, int(self.compression * tensor.nelement()))
        m = tensor.view(tensor.shape[0], -1)
        size = self.rank * (m.shape[0] + m.shape[1])
        return min(size, tensor.nelement())

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        values_list = []
        start_idx_list = []
        block_sizes = []

        with self.timer("reduce.block", verbosity=2):
            for tensor in grad_in:
                block_size = self.block_size(tensor)
                block_sizes.append(block_size)
                if block_size == tensor.nelement():
                    start_idx = 0
                else:
                    start_idx = self.rng.choice(tensor.nelement())
                start_idx_list.append(start_idx)
                end_idx = min(start_idx + block_size, tensor.nelement())
                bfr = torch.empty(block_size, dtype=torch.float32, device=self.device)
                bfr[: end_idx - start_idx] = tensor.view(-1)[start_idx:end_idx]
                rest = block_size - (end_idx - start_idx)
                if rest > 0:
                    bfr[end_idx - start_idx :] = tensor.view(-1)[:rest]
                values_list.append(bfr)

        with self.timer("reduce.flatpack", verbosity=2):
            flat_values = TensorBuffer(values_list)

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, start_idx, block_size in zip(grad_in, memory_out, start_idx_list, block_sizes):
                end_idx = min(start_idx + block_size, tensor.nelement())
                rest = block_size - (end_idx - start_idx)
                mem.data = tensor.clone()
                mem.view(-1)[start_idx:end_idx] = 0.0
                if rest > 0:
                    mem.view(-1)[:rest] = 0.0

        with self.timer("reduce.reduce", verbosity=2):
            flat_values.all_reduce()
            flat_values.buffer /= self.n_workers
            bits_communicated += flat_values.bits()

        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, start_idx, block_size, values in zip(grad_in, grad_out, start_idx_list, block_sizes, flat_values):
                end_idx = min(start_idx + block_size, tensor.nelement())
                rest = block_size - (end_idx - start_idx)
                out.data.zero_()
                out.view(-1)[start_idx:end_idx] = values[: end_idx - start_idx]
                if rest > 0:
                    out.view(-1)[:rest] = values[end_idx - start_idx :]

        return bits_communicated


class RandomSparseReducer(Reducer):
    def __init__(self, random_seed, device, timer, rank):
        super().__init__(random_seed, device, timer)
        self.rank = rank

    def block_size(self, tensor):
        # return max(1, int(self.compression * tensor.nelement()))
        m = tensor.view(tensor.shape[0], -1)
        size = self.rank * (m.shape[0] + m.shape[1])
        return min(size, tensor.nelement())

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        indices_list = []
        values_list = []

        with self.timer("reduce.block", verbosity=2):
            for tensor in grad_in:
                block_size = self.block_size(tensor)
                if block_size == tensor.nelement():
                    indices = np.arange(tensor.nelement())
                else:
                    indices = self.rng.choice(tensor.nelement(), block_size, replace=False)
                indices_list.append(indices)
                values = tensor.view(-1)[indices]
                values_list.append(values)

        with self.timer("reduce.flatpack", verbosity=2):
            flat_values = TensorBuffer(values_list)

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, indices in zip(grad_in, memory_out, indices_list):
                mem.data[:] = tensor
                mem.view(-1)[indices] = 0.0

        with self.timer("reduce.reduce", verbosity=2):
            flat_values.all_reduce()
            flat_values.buffer.data /= self.n_workers
            bits_communicated += flat_values.bits()

        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, values, indices in zip(grad_in, grad_out, flat_values, indices_list):
                out.data.zero_()
                out.view(-1)[indices] = values

        return bits_communicated


class SVDReducer(Reducer):
    def __init__(self, random_seed, device, timer, rank=1):
        super().__init__(random_seed, device, timer)
        self.rank = rank

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        # Split the tensors into rank1-ones that will be reduced un-compressed
        # and rank > 1 tensors that are compressed
        rank1_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() <= 1
        ]
        high_rank_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() > 1
        ]

        # Communicate rank 1 tensors
        bits_communicated += self._reduce_rank1(rank1_tensors)

        for tensor, out, mem in high_rank_tensors:
            m = tensor.shape[0]
            n = tensor.nelement() // m

            matrix = tensor.view(tensor.shape[0], -1)

            rnk = min(self.rank, m, n)
            u, s, v = torch.svd(matrix)
            u, s, v = u[:, :rnk], s[:rnk], v[:, :rnk]

            mem.data[:] = tensor
            mem.view(*matrix.shape).data -= torch.einsum("in, n, jn -> ij", u, s, v)

            if self.n_workers > 1:
                worker_u = [torch.empty_like(u) for i in range(self.n_workers)]
                worker_v = [torch.empty_like(v) for i in range(self.n_workers)]
                worker_s = [torch.empty_like(s) for i in range(self.n_workers)]
                h1 = all_gather(worker_u, u, async_op=True)
                h2 = all_gather(worker_v, v, async_op=True)
                h3 = all_gather(worker_s, s, async_op=True)
                h1.wait()
                h2.wait()
                h3.wait()
            else:
                worker_u = [u]
                worker_v = [v]
                worker_s = [s]

            out.data[:] = 0.0
            for uu, ss, vv in zip(worker_u, worker_s, worker_v):
                out.view(*matrix.shape).add_(
                    1.0 / self.n_workers, torch.einsum("in, n, jn -> ij", uu, ss, vv)
                )

            bits_communicated += n_bits(u) + n_bits(s) + n_bits(v)

        return bits_communicated

    def _reduce_rank1(self, pairs):
        with self.timer("reduce.rank1.zero_memory", verbosity=2):
            for _, _, mem in pairs:
                mem.zero_()

        list_in = [tensor for (tensor, _, _) in pairs]
        list_out = [out for (_, out, _) in pairs]

        with self.timer("reduce.rank1.reduce", verbosity=2):
            return reduce_mean_list(self.device, list_in, list_out, self.timer)

class RankKReducer(Reducer):
    def __init__(self, random_seed, device, timer, n_power_iterations=0, reuse_query=False, rank=1):
        super().__init__(random_seed, device, timer)
        assert n_power_iterations == 0
        self.rank = rank
        self.p_memory = None
        self.q_memory = None
        self.reuse_query = reuse_query

    def set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        # orthogonalize(vector)

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        torch.cuda.synchronize()
        # start_epoch = torch.cuda.Event(enable_timing=True)
        # stop_epoch = torch.cuda.Event(enable_timing=True)
        # Split the tensors into rank1-ones that will be reduced un-compressed
        # and rank > 1 tensors that are compressed
        rank1_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() <= 1
        ]
        high_rank_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() > 1
        ]

        # We are building a rank-1 approximation of every tensor
        # that can be interpreted as a matrix. Let the approximation be
        # M = p q^T
        # We are allocating consequtive memory for the p's and q's

        memory_is_uninitialized = self.p_memory is None
        # start_epoch.record()
        with self.timer("reduce.allocate_memory", verbosity=2):
            p_total_size = 0
            q_total_size = 0
            for tensor, _, _ in high_rank_tensors:
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)
                p_total_size += n * rank
                q_total_size += m * rank
            if self.p_memory is None:
                self.p_memory = torch.empty(p_total_size, device=self.device)
                self.q_memory = torch.empty(q_total_size, device=self.device)

            # Find them again and make lists of pointers
            ps = []
            qs = []
            p_idx = 0
            q_idx = 0
            for tensor, _, _ in high_rank_tensors:
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)
                ps.append(self.p_memory[p_idx : p_idx + n * rank].view(n, rank))
                qs.append(self.q_memory[q_idx : q_idx + m * rank].view(m, rank))
                p_idx += n * rank
                q_idx += m * rank
        # stop_epoch.record()
        # torch.cuda.synchronize()
        # print ("Time spent allocate memory{}".format(start_epoch.elapsed_time(stop_epoch)))
        # start_epoch.record()
        with self.timer("reduce.prepare.q", verbosity=2):
            for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape

                if self.reuse_query and not memory_is_uninitialized:
                    # orthogonalize(q)
                    pass
                else:
                    # Sample a query vector q
                    self.set_random(q)
        # stop_epoch.record()
        # torch.cuda.synchronize()
        # print ("Time spent prepare q{}".format(start_epoch.elapsed_time(stop_epoch)))
        # start_epoch.record()
        with self.timer("reduce.compute.p", verbosity=2):
            for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
                matrix = tensor.view(tensor.shape[0], -1)
                torch.matmul(matrix, q, out=p)
        # stop_epoch.record()
        # torch.cuda.synchronize()
        # print ("Time spent compute p{}".format(start_epoch.elapsed_time(stop_epoch)))
        # start_epoch.record()
        with self.timer("reduce.p", verbosity=2):
            all_reduce(self.p_memory)
        # stop_epoch.record()
        # torch.cuda.synchronize()
        # print ("P_memory shape {}".format(self.p_memory.shape))
        bits_communicated += n_bits(self.p_memory)
        # print ("Time spent reduce p{}".format(start_epoch.elapsed_time(stop_epoch)))
        # Start communicating rank 1 tensors
        # start_epoch.record()
        with self.timer("reduce.rank1.pack", verbosity=2):
            rank1_tensor_list = TensorBuffer([tensor for (tensor, _, _) in rank1_tensors])
        with self.timer("reduce.rank1.all_reduce", verbosity=2):
            rank1_handle = rank1_tensor_list.all_reduce(async_op=True)
            bits_communicated += rank1_tensor_list.bits()
        # stop_epoch.record()
        # torch.cuda.synchronize()
        # print("Time spent rank1 reduce{}".format(
            # start_epoch.elapsed_time(stop_epoch)))
        # start_epoch.record()
        with self.timer("reduce.normalize.p", verbosity=2):
            for p in ps:
                orthogonalize(p)
        # stop_epoch.record()
        # torch.cuda.synchronize()
        # print ("Time spent normalize p{}".format(start_epoch.elapsed_time(stop_epoch)))
        # start_epoch.record()
        with self.timer("reduce.compute.q", verbosity=2):
            for p, q, (tensor, _, _) in zip(ps, qs, high_rank_tensors):
                matrix = tensor.view(tensor.shape[0], -1)
                torch.matmul(matrix.t(), p, out=q)
        # stop_epoch.record()
        # torch.cuda.synchronize()
        # print ("Time spent compute q{}".format(start_epoch.elapsed_time(stop_epoch)))
        # start_epoch.record()
        with self.timer("reduce.q", verbosity=2):
            all_reduce(self.q_memory)
            # print ("Q_memory shape {}".format(self.q_memory.shape))
            bits_communicated += n_bits(self.q_memory)
            self.q_memory.data[:] /= self.n_workers
        # stop_epoch.record()
        # torch.cuda.synchronize()
        # print ("time spent reduce q{}".format(start_epoch.elapsed_time(stop_epoch)))
        # start_epoch.record()
        with self.timer("reduce.outerprod", verbosity=2):
            for p, q, (tensor, out, mem) in zip(ps, qs, high_rank_tensors):
                # Set the output gradient
                torch.matmul(p, q.t(), out=out.data[:])
                mem.data[:] = tensor - out
        # stop_epoch.record()
        # torch.cuda.synchronize()
        # print("time spent outerprod{}".format(start_epoch.elapsed_time(stop_epoch)))
        # start_epoch.record()
        with self.timer("reduce.rank1.unpack", verbosity=2):
            rank1_handle.wait()
            rank1_tensor_list.buffer /= self.n_workers
            rank1_tensor_list.unpack([out for (_, out, _) in rank1_tensors])
        # stop_epoch.record()
        # torch.cuda.synchronize()
        # print("time spent rank 1 unpack{}".format(start_epoch.elapsed_time(stop_epoch)))
        # print ("\n")
        # print (bits_communicated)
        return bits_communicated

def list_to_tensor(input_list):
    temp_list = [t.reshape(-1) for t in input_list]
    return (torch.cat(temp_list))


class MsTopKReducer(Reducer):
    """
    This is the statistical topK reducer which performs the approximate topK
    """
    def __init__(self, random_seed, device, timer, k):
        super().__init__(random_seed, device, timer)
        self.k = k
        self.N = 100
    def reduce(self, grad_in, grad_out):
        start_time = torch.cuda.Event(enable_timing=True)
        stop_time = torch.cuda.Event(enable_timing=True)
        start.record()
        grad_in = list_to_tensor(grad_in)
        grad_1d = grad_in.reshape(-1) #reshaping to 1d
        k = int(self.k*len(grad_1d)) #change percent to actual number
        a = torch.abs(grad_1d)
        a_hat = torch.mean(a)
        u = torch.max(a)
        l = 0
        r = 1
        k1 = 0
        k2 = len(grad_1d)
        thres1 = 0
        thres2 = 0
        for i in range(self.N):
            ratio = l + (r-l)/2
            thres = a_hat + ratio*(u-a_hat)
            nnz = torch.count_nonzero(a >= thres)
            if nnz <= k:
                r = ratio
                if nnz > k1:
                    k1 = nnz
                    thres1 = thres
            elif nnz > k:
                l= ratio
                if nnz < k2:
                    k2 = nnz
                    thres2 = thres
        l1 = torch.nonzero(a>= thres1, as_tuple=True)[0] #since 1d no problem
        l2 = torch.nonzero((a<thres1) & (a >= thres2), as_tuple=True)[0]

        if len(l2)-(k-k1)+1 < 0:
            l = torch.cat((l1, l2[0:k-len(l1)]))
            # print("Fake if")
        else:
            rand = random.randint(0, len(l2)-(k-k1)+1)
            l = torch.cat((l1, l2[rand:rand+k-k1]))
            # print ("Actual if")
        kai = grad_1d[l]
        stop_time.record()
        print ("Time taken {}".format(start_time.elapsed_time(stop_time)))
        kai = torch.zeros((k), device=kai.device, dtype=kai.dtype)
        l = torch.zeros((k), device=l.device, dtype=l.dtype)
        index_list = [torch.zeros_like(l, device=l.device, dtype=l.dtype) for _ in range(self.n_workers)]
        ind_wait = torch.distributed.all_gather(index_list, l, async_op=True)
        value_list = [torch.zeros_like(kai, device=kai.device, dtype=kai.dtype) for _ in range(self.n_workers)]
        val_wait = torch.distributed.all_gather(value_list, kai, async_op=True)
        ind_wait.wait()
        val_wait.wait()

        grad_accum = torch.zeros_like(grad_1d)
        for idx, vals in zip(index_list, value_list):
            grad_accum[idx] += vals
        
        # print ("Starting element copy") 
        # start_index = 0
        # for idx, ts in enumerate(grad_out):
            # num_element_ts = grad_out[idx].numel()
            # ts = grad_accum[start_index:start_index+num_element_ts]
            # grad_out[idx] = ts
            # start_index += num_element_ts
        # print ("Done element copy")
        # grad_out = grad_1.reshape(grad_in.shape)
        return grad_out




class HalfRankKReducer(Reducer):
    """
    This is an adapted version of RankKReducer that
    only does one matrix multiplication per iteration
    """

    def __init__(self, random_seed, device, timer, rank=1):
        super().__init__(random_seed, device, timer)
        self.rank = rank
        self.p_memory = None
        self.q_memory = None
        self.next_operation = "p"  # or q, binary state

    def set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        orthogonalize(vector)

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        # Split the tensors into rank1-ones that will be reduced un-compressed
        # and rank > 1 tensors that are compressed
        rank1_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() <= 1
        ]
        high_rank_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() > 1
        ]

        # Communicate rank 1 tensors
        with self.timer("reduce.rank1.pack", verbosity=2):
            rank1_tensor_list = TensorBuffer([tensor for (tensor, _, _) in rank1_tensors])
        with self.timer("reduce.rank1.all_reduce", verbosity=2):
            rank1_handle = rank1_tensor_list.all_reduce(async_op=True)
            bits_communicated += rank1_tensor_list.bits()

        # We are building a rank-1 approximation of every tensor
        # that can be interpreted as a matrix. Let the approximation be
        # M = p q^T
        # We are allocating consequtive memory for the p's and q's

        memory_is_uninitialized = self.p_memory is None

        if self.p_memory is None:
            with self.timer("reduce.allocate_memory", verbosity=2):
                p_total_size = 0
                q_total_size = 0
                for tensor, _, _ in high_rank_tensors:
                    matrix = tensor.view(tensor.shape[0], -1)
                    n, m = matrix.shape
                    rank = min(n, m, self.rank)
                    p_total_size += n * rank
                    q_total_size += m * rank
                self.p_memory = torch.empty(p_total_size, device=self.device)
                self.q_memory = torch.empty(q_total_size, device=self.device)

        with self.timer("reduce.build_index", verbosity=2):
            ps = []
            qs = []
            p_idx = 0
            q_idx = 0
            for tensor, _, _ in high_rank_tensors:
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)
                ps.append(self.p_memory[p_idx : p_idx + n * rank].view(n, rank))
                qs.append(self.q_memory[q_idx : q_idx + m * rank].view(m, rank))
                p_idx += n * rank
                q_idx += m * rank

        if self.next_operation == "p":
            self.next_operation = "q"
            with self.timer("reduce.normalize.q", verbosity=2):
                for q in qs:
                    if memory_is_uninitialized:
                        self.set_random(q)
                    else:
                        orthogonalize(q)

            with self.timer("reduce.compute.p", verbosity=2):
                for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
                    matrix = tensor.view(tensor.shape[0], -1)
                    torch.matmul(matrix, q, out=p)

            with self.timer("reduce.fill_memory"):
                for p, q, (tensor, _, mem) in zip(ps, qs, high_rank_tensors):
                    matrix = tensor.view(tensor.shape[0], -1)
                    # Keep what we couldn't send in memory
                    mem.data[:] = (matrix - torch.einsum("nr, mr -> nm", (p, q))).view(
                        *tensor.shape
                    )

            with self.timer("reduce.p", verbosity=2):
                all_reduce(self.p_memory)
                bits_communicated += n_bits(self.p_memory)
                self.p_memory.data[:] /= self.n_workers

        elif self.next_operation == "q":
            self.next_operation = "p"
            with self.timer("reduce.normalize.p", verbosity=2):
                for p in ps:
                    orthogonalize(p)

            with self.timer("reduce.compute.q", verbosity=2):
                for p, q, (tensor, _, _) in zip(ps, qs, high_rank_tensors):
                    matrix = tensor.view(tensor.shape[0], -1)
                    torch.matmul(matrix.t(), p, out=q)

            with self.timer("reduce.fill_memory", verbosity=2):
                for p, q, (tensor, _, mem) in zip(ps, qs, high_rank_tensors):
                    matrix = tensor.view(tensor.shape[0], -1)
                    # Keep what we couldn't send in memory
                    mem.data[:] = (matrix - torch.einsum("nr, mr -> nm", (p, q))).view(
                        *tensor.shape
                    )

            with self.timer("reduce.q", verbosity=2):
                all_reduce(self.q_memory)
                bits_communicated += n_bits(self.q_memory)
                self.q_memory.data[:] /= self.n_workers

        with self.timer("reduce.outerprod", verbosity=2):
            for p, q, (tensor, out, _) in zip(ps, qs, high_rank_tensors):
                # Set the output gradient
                out.data[:] = torch.einsum("nr, mr -> nm", (p, q)).view(*tensor.shape)

        with self.timer("reduce.rank1.unpack", verbosity=2):
            rank1_handle.wait()
            rank1_tensor_list.buffer /= self.n_workers
            rank1_tensor_list.unpack([out for (_, out, _) in rank1_tensors])

        return bits_communicated


# def orthogonalize(matrix):
#     # This is super slow
#     r = torch.empty(1, device=matrix.device)  # dummy memory, we don't care about r
#     torch.qr(matrix, out=(matrix, r))
#     del r

@torch.jit.script
def orthogonalize(matrix):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col


class ExactReducer(Reducer):
    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        # with self.timer("reduce.zero_mem", verbosity=2):
            # for mem in memory_out:
                # mem.zero_()

        with self.timer("reduce.build_lists", verbosity=2):
            list_in = grad_in
            list_out = grad_out

        with self.timer("reduce.reduce", verbosity=2):
            bits_communicated = reduce_mean_list(self.device, list_in, list_out, self.timer)

        return bits_communicated

class ExactReducerPs(Reducer):
    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        # with self.timer("reduce.zero_mem", verbosity=2):
            # for mem in memory_out:
                # mem.zero_()

        with self.timer("reduce.build_lists", verbosity=2):
            list_in = grad_in
            list_out = grad_out
        

        tensor_to_reduce = torch.cat([l.view(-1) for l in list_in])

        receive_tensor = torch.zeros_like(tensor_to_reduce)
        
        if self.rank == 0:
            for i in range(1, self.n_workers):
                torch.distributed.recv(receive_tensor, src=i)
                tensor_to_reduce += receive_tensor
        else:
            torch.distributed.send(tensor_to_reduce, 0)


        # torch.distributed.barrier()

        torch.distributed.broadcast(tensor_to_reduce, src=0)
        
        # torch.distributed.barrier()

        # with self.timer("reduce.reduce", verbosity=2):
            # bits_communicated = reduce_mean_list(self.device, list_in, list_out, self.timer)

        return None 



def reduce_mean_list(
    device: torch.device, list_in: List[torch.Tensor], list_out: List[torch.Tensor], timer
):
    if torch.distributed.is_available():
        n_workers = torch.distributed.get_world_size()
    else:
        n_workers = 1

    if n_workers == 1:
        for t_in, t_out in zip(list_in, list_out):
            t_out[:] = t_in
        return 0

    with timer("reduce.mean.pack"):
        buffer = TensorBuffer(list_in)

    with timer("reduce.mean.allreduce"):
        buffer.all_reduce()
        buffer.buffer /= n_workers
        bits_communicated = buffer.bits()

    with timer("reduce.mean.unpack", verbosity=2):
        buffer.unpack(list_out)

    return bits_communicated


def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()


class TensorBuffer():
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """
    def __init__(self, tensors):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors = tensors

        self.buffer = torch.cat([t.view(-1) for t in tensors]) # copies
    
    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(*self._tensors[index].shape)

    def __len__(self):
        return len(self._tensors)

    def pack(self, tensors=None):
        # Optional. init already does this.
        if tensors is None:
            tensors = self._tensors
        for tensor, entry in zip(tensors, self):
            entry[:] = tensor

    def unpack(self, tensors):
        for tensor, entry in zip(tensors, self):
            tensor[:] = entry

    def nelement(self):
        return self.buffer.nelement()

    def element_size(self):
        return self.buffer.element_size()

    def bits(self):
        return 8 * self.nelement() * self.element_size()

    def all_reduce(self, async_op=False):
        return torch.distributed.all_reduce(self.buffer, async_op=async_op)
    
    def all_gather(self, async_op=False):
        n_workers = torch.distributed.get_world_size() if torch.distributed.is_available() else 1
        buffers = [torch.empty_like(self.buffer) for i in range(n_workers)]
        handle = all_gather(buffers, self.buffer, async_op=async_op)
        if async_op:
            return buffers, handle
        else:
            return buffers
    

def all_reduce(*args, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_reduce(*args, **kwargs)


def all_gather(out_list, in_tensor, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_gather(out_list, in_tensor, **kwargs)
    else:
        assert len(out_list) == 1
        out_list[0].data = in_tensor


@torch.jit.script
def l2norm(x):
    return torch.sqrt(torch.sum(x ** 2))


def normalize_(tensor):
    """Divide by L2 norm. In place"""
    tensor /= l2norm(tensor)


class SignCompressor:
    """Taken from https://github.com/PermiJW/signSGD-with-Majority-Vote"""

    def packing(self, src_tensor):
        src_tensor = torch.sign(src_tensor)
        src_tensor_size = src_tensor.size()
        src_tensor = src_tensor.view(-1)
        src_len = len(src_tensor)
        add_elm = 32 - (src_len % 32)
        if src_len % 32 == 0:
            add_elm = 0
        new_tensor = torch.zeros([add_elm], dtype=torch.float32, device=src_tensor.device)
        src_tensor = torch.cat((src_tensor, new_tensor), 0)
        src_tensor = src_tensor.view(32, -1)
        src_tensor = src_tensor.to(dtype=torch.int32)
        dst_tensor = bit2byte.packing(src_tensor)
        dst_tensor = dst_tensor.to(dtype=torch.int32)
        return dst_tensor, src_tensor_size

    def unpacking(self, src_tensor, src_tensor_size):
        src_element_num = self.element_num(src_tensor_size)
        add_elm = 32 - (src_element_num % 32)
        if src_element_num % 32 == 0:
            add_elm = 0
        src_tensor = src_tensor.int()
        new_tensor = torch.ones(
            src_element_num + add_elm, device=src_tensor.device, dtype=torch.int32
        )
        new_tensor = new_tensor.view(32, -1)
        new_tensor = bit2byte.unpacking(src_tensor, new_tensor)
        new_tensor = new_tensor.view(-1)
        new_tensor = new_tensor[:src_element_num]
        new_tensor = new_tensor.view(src_tensor_size)
        new_tensor = -new_tensor.add_(-1)
        new_tensor = new_tensor.float()
        return new_tensor

    def majority_vote(self, src_tensor_list):
        voter_num = len(src_tensor_list)
        src_tensor = torch.stack(src_tensor_list)
        src_tensor = src_tensor.view(-1)
        full_size = 32 * len(src_tensor)
        new_tensor = torch.ones(full_size, device=src_tensor.device, dtype=torch.int32)
        new_tensor = new_tensor.view(32, -1)
        new_tensor = bit2byte.unpacking(src_tensor, new_tensor)
        new_tensor = -new_tensor.add_(-1)
        # sum
        new_tensor = new_tensor.permute(1, 0).contiguous().view(voter_num, -1)
        new_tensor = torch.sum(new_tensor, 0)
        new_tensor = new_tensor.view(-1, 32).permute(1, 0)
        new_tensor = torch.sign(new_tensor)
        new_tensor = bit2byte.packing(new_tensor)
        new_tensor = new_tensor.to(dtype=torch.int32)
        return new_tensor

    def element_num(self, size):
        num = 1
        for i in range(len(size)):
            num *= size[i]
        return num

    def compress(self, src_tensor):
        return self.packing(src_tensor)

    def uncompress(self, src_tensor, src_tensor_size):
        dst_tensor = self.unpacking(src_tensor, src_tensor_size)
        return dst_tensor
