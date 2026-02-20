# autopipe.py
import functools
from collections import deque
from typing import Dict, List, Tuple

COMM_OVERHEAD = 0
MAX_INT64 = 2**63 - 1
MAX_INT32 = 2**31 - 1
INF = 10**18


def _prefix_sum_and_dp(
    model: List[int],
    num_stages: int,
    block_time: List[List[int]],
    prefix_sum: List[int],
    dp: List[List[int]],
):
    """C++ calculate_prefix_sum_and_dp"""
    num_blocks = len(model)
    max_parts = min(num_blocks, num_stages)

    # sum prefix
    prefix_sum.clear()
    prefix_sum.append(0)
    for b in model:
        t = block_time[0][b] + block_time[1][b]
        prefix_sum.append(prefix_sum[-1] + t)

    # DP sheet
    dp.clear()
    for _ in range(num_blocks + 1):
        dp.append([MAX_INT64] * (max_parts + 1))
    dp[0][0] = 0

    for blocks in range(1, num_blocks + 1):
        max_p = min(blocks, max_parts)
        for parts in range(1, max_p + 1):
            best = MAX_INT64
            for prev in range(blocks):
                cur = max(dp[prev][parts - 1], prefix_sum[blocks] - prefix_sum[prev])
                best = min(best, cur)
                if best == 0:
                    break
            dp[blocks][parts] = best


def _reconstruct(
    model: List[int],
    prefix_sum: List[int],
    dp: List[List[int]],
    rem_blocks: int,
    rem_parts: int,
    out: List[List[int]],
):
    """C++ reconstruct_partitions"""
    if rem_blocks == 0 and rem_parts == 0:
        return
    if rem_blocks <= 0 or rem_parts <= 0 or rem_blocks < rem_parts:
        raise RuntimeError("Error during partition reconstruction")

    prev_end = 0
    while prev_end < rem_blocks:
        lhs = dp[prev_end][rem_parts - 1]
        rhs = prefix_sum[rem_blocks] - prefix_sum[prev_end]
        if dp[rem_blocks][rem_parts] == max(lhs, rhs):
            break
        prev_end += 1

    chunk = [model[i] for i in range(prev_end, rem_blocks)]
    out.append(chunk)
    _reconstruct(model, prefix_sum, dp, prev_end, rem_parts - 1, out)


def _block_partition_algo(
    model: List[int], num_stages: int, block_time: List[List[int]]
) -> List[List[int]]:
    """C++ block_partition_algorithm"""
    prefix_sum: List[int] = []
    dp: List[List[int]] = []
    _prefix_sum_and_dp(model, num_stages, block_time, prefix_sum, dp)

    parts: List[List[int]] = []
    _reconstruct(model, prefix_sum, dp, len(model), num_stages, parts)
    parts.reverse()
    return parts


# ---------- Time Cal ----------
def _calc_stage_times(
    partition: List[List[int]],
    block_time: List[List[int]],
    fwd: List[int],
    bwd: List[int],
    last_mb: List[int],
):
    """C++ calculate_stage_times"""
    num_stages = len(partition)
    num_micro = num_stages * 2
    for i in range(num_stages):
        last_mb[i] = num_micro - num_stages + i

    for i in range(1, num_stages + 1):
        fwd_sum = sum(block_time[0][b] for b in partition[i - 1])
        bwd_sum = sum(block_time[1][b] for b in partition[i - 1])
        fwd[i] = fwd_sum
        bwd[i] = bwd_sum


def _steady_phase(
    last_mb: List[int], fwd: List[int], bwd: List[int]
) -> Tuple[int, int]:
    """C++ calculate_steady_phase"""
    num_stages = len(last_mb)
    num_micro = num_stages * 2

    dp = [[[0, 0] for _ in range(num_micro)] for __ in range(num_stages + 2)]

    # init
    init_bwd = 0
    for s in range(num_stages):
        init_bwd += fwd[s + 1]
        if s != num_stages - 1:
            init_bwd += COMM_OVERHEAD
    for s in range(num_stages - 1, -1, -1):
        dp[s + 1][0][0] = MAX_INT64
        dp[s + 1][0][1] = init_bwd
        init_bwd += bwd[s + 1] + COMM_OVERHEAD

    for mb in range(1, num_micro):
        # forward
        for s in range(num_stages):
            if mb <= last_mb[s]:
                val = max(dp[s][mb - 1][0] + fwd[s], dp[s + 1][mb - 1][1] + bwd[s + 1])
                if s != 0:
                    val += COMM_OVERHEAD
                dp[s + 1][mb][0] = val
        # backward
        for s in range(num_stages - 1, -1, -1):
            if mb <= last_mb[s]:
                val = max(dp[s + 2][mb][1] + bwd[s + 2], dp[s + 1][mb][0] + fwd[s + 1])
                if s != num_stages - 1:
                    val += COMM_OVERHEAD
                dp[s + 1][mb][1] = val

    # find critical path
    critical = num_stages - 1
    while critical >= 0:
        ok = True
        for mb in range(1, last_mb[critical] + 1):
            fcomm = COMM_OVERHEAD if critical != 0 else 0
            bcomm = COMM_OVERHEAD if critical != num_stages - 1 else 0
            if (
                dp[critical + 1][mb][0]
                != dp[critical + 1][mb - 1][1] + bwd[critical + 1] + fcomm
            ):
                ok = False
                break
            if (
                dp[critical + 1][mb][1]
                != dp[critical + 1][mb][0] + fwd[critical + 1] + bcomm
            ):
                ok = False
                break
        if ok:
            break
        critical -= 1

    if critical < 0:
        # Backstop: The stage that finishes the last micro-batch is the critical stage.
        _, best_time = 0, -1
        for s in range(num_stages):
            t = dp[s + 1][last_mb[s]][1]  # Finished time of the last backward
            if t > best_time:
                best_time, critical = t, s
    return dp[critical + 1][last_mb[critical]][0], critical


def _cooldown(
    num_stages: int, critical: int, last_fwd_start: int, fwd: List[int], bwd: List[int]
) -> int:
    """C++ calculate_cooldown_phase"""
    sz = num_stages - critical
    if sz <= 0:
        return last_fwd_start

    dp = [[0] * sz for _ in range(sz)]
    bwd_start = last_fwd_start
    for i in range(sz):
        bwd_start += fwd[critical + 1 + i]
        if critical + i != num_stages - 1:
            bwd_start += COMM_OVERHEAD
        dp[i][sz - 1 - i] = bwd_start

    for col in range(sz - 2, -1, -1):
        for row in range(sz - col - 2, -1, -1):
            o1 = dp[row][col + 1] + bwd[critical + 1 + row] + COMM_OVERHEAD
            o2 = dp[row + 1][col] + bwd[critical + 1 + row + 1] + COMM_OVERHEAD
            dp[row][col] = max(o1, o2)
            if row > 0:
                dp[row][col] = max(dp[row][col], dp[row - 1][col + 1])
    return dp[0][0]


def _training_time(
    partition: List[List[int]], block_time: List[List[int]]
) -> Tuple[int, int]:
    """C++ calculate_training_time"""
    num_stages = len(partition)
    last_mb = [0] * num_stages
    fwd = [0] * (num_stages + 2)
    bwd = [0] * (num_stages + 2)

    # 计算阶段时间
    for i in range(num_stages):
        last_mb[i] = num_stages * 2 - num_stages + i
        fwd[i + 1] = sum(block_time[0][b] for b in partition[i])
        bwd[i + 1] = sum(block_time[1][b] for b in partition[i])

    steady_time, critical = _steady_phase(last_mb, fwd, bwd)
    if steady_time == MAX_INT64:
        raise RuntimeError("Failed to calculate steady phase")

    last_bwd_start = _cooldown(num_stages, critical, steady_time, fwd, bwd)
    flush = last_bwd_start
    for stage in range(critical, 0, -1):
        flush += bwd[stage + 1] + COMM_OVERHEAD
    flush += bwd[1]
    return flush, critical


# ---------- 最优分区搜索 ----------
def _find_best(
    block_time: List[List[int]],
    num_stages: int,
    init_partition: List[List[int]],
    prefix_sum: List[int],
    dp: List[List[int]],
) -> Dict:
    """
    C++ find_best_partition
    return: {"partition": [[...], ...], "cost": int, "critical_stage": int}
    """

    # Hash func: C++ VectorHash
    @functools.lru_cache(maxsize=None)
    def _hash(p):
        h = 0
        for inner in p:
            for v in inner:
                h ^= (v + 0x9E3779B9) + (h << 6) + (h >> 2)
        return h

    # C++ VectorEqual
    def _eq(a, b):
        return len(a) == len(b) and all(
            len(ai) == len(bi) and all(av == bv for av, bv in zip(ai, bi))
            for ai, bi in zip(a, b)
        )

    visited = set()
    queue = deque([init_partition])
    visited.add(_hash(tuple(tuple(r) for r in init_partition)))

    # Best result
    best = {"partition": init_partition, "cost": INF, "critical_stage": MAX_INT32}

    while queue:
        cur = queue.popleft()

        # Current time of partition.
        last_mb = [0] * num_stages
        fwd = [0] * (num_stages + 2)
        bwd = [0] * (num_stages + 2)
        _calc_stage_times(cur, block_time, fwd, bwd, last_mb)
        cost, critical = _training_time(cur, block_time)

        # update best
        if cost < best["cost"]:
            best = {"partition": cur, "cost": cost, "critical_stage": critical}

        # If the critical path is not in the first segment,
        # try re-partitioning all the blocks before the critical stage.
        if critical > 0:
            # Collect all blocks before (and including the first block of) the critical stage.
            blocks_before = []
            for stage in range(critical):
                blocks_before.extend(cur[stage])
            blocks_before.append(cur[critical][0])

            # Redo the critical stage partitioning for these blocks with the same DP.
            model_before = blocks_before
            new_parts: List[List[int]] = []
            _reconstruct(
                model_before, prefix_sum, dp, len(model_before), critical, new_parts
            )
            new_parts.reverse()
            blocks_before.pop()

            full_new = new_parts
            # Put the remaining blocks of the critical segment back.
            if len(full_new) <= critical:
                full_new.append([])
            full_new[critical].extend(cur[critical][1:])

            for stage in range(critical + 1, len(cur)):
                full_new.append(cur[stage])

            # "Hash deduplication
            key = _hash(tuple(tuple(r) for r in full_new))
            if key not in visited:
                visited.add(key)
                queue.append(full_new)

    return best


# ---------- main ----------
def auto_partition(
    forward_times: List[int], backward_times: List[int], num_stages: int
) -> List[int]:

    if not forward_times or not backward_times:
        raise ValueError("Input vectors cannot be empty")
    if len(forward_times) != len(backward_times):
        raise ValueError("Forward and backward vectors must have same size")
    if num_stages <= 0 or num_stages > len(forward_times):
        raise ValueError("Invalid number of pipeline stages")

    block_time = [forward_times, backward_times]
    model = list(range(len(forward_times)))

    init_partition = _block_partition_algo(model, num_stages, block_time)
    prefix_sum: List[int] = []
    dp: List[List[int]] = []
    _prefix_sum_and_dp(model, num_stages, block_time, prefix_sum, dp)

    best = _find_best(block_time, num_stages, init_partition, prefix_sum, dp)

    result = [stage[0] for stage in best["partition"]]
    return result


if __name__ == "__main__":
    import traceback

    try:
        fwd_flops = [10, 20, 30, 15, 25]  # five block
        bwd_flops = [10, 20, 30, 15, 25]
        for stages in 1, 2, 3, 4, 5:
            test_out = auto_partition(fwd_flops, bwd_flops, stages)
            print(f"stages={stages}, result={test_out}, len={len(test_out)}")

    except Exception as e:
        traceback.print_exc()
