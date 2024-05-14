# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flight Recorder Trace Analyzer

This script primarily merges data from individual flight recorder buffers from individual ranks in a
PyTorch Distributed program into a flattened database format that can be used for further analysis.

However as part of the merging process, it is necessary to perform some analysis in order to match operators
on one rank with corresponding operators on other ranks and register them as one 'collective' entry.  During this
process, a significant amount of useful information can already be extracted such as where the first mismatch occurs
in cases of desync (when not all ranks issue a compatible collective in a particular process group).


Not Yet Implemented
- TODO- tracebacks aren't implemented

Known Issues
- Flight Recorder buffer sequence_id information is not sufficient to match collectives and coalseced collectives
  unless we have the trace data from the beginning of the program.  To enable confident analysis of trace buffers that
  do not start from zero (and to simplify the script's matching logic) we need to add more information to the recorder.
- Currently, the script omits checking the 'status' of collectives.  We can look for the first 'non completed'
  collective easily enough and report that.

Usage
python fr_trace.py -d <dump dir containing trace files> [-o <output file>]

- Omitting the optional output file will still yield analysis information to stdout
- the output file is a pickle of the flat DB, which may change in format in the future.
"""

import argparse
import os
import pickle
from hashlib import sha256
from typing import _eval_type, Generic, List, NamedTuple, Tuple, Type, TypeVar

from tabulate import tabulate

T = TypeVar("T")


class Ref(Generic[T]):
    pass


class TypeInfo(NamedTuple):
    name: str
    fields: List[Tuple[str, Type]]

    @classmethod
    def from_type(cls, c):
        return cls(
            c.__name__,
            [
                (f, _eval_type(c.__annotations__[f], globals(), {}, frozenset()))
                for f in c._fields
            ],
        )


"""
Stacktrace cache
TODO
"""


"""
Collective Matching logic
"""
COLLECTIVES = {
    "broadcast",
    "all_gather",
    "all_reduce",
    "_all_gather_base",
    "all_gather_into_tensor_coalesced",
    "reduce_scatter_tensor_coalesced",
    "gather",
    "scatter",
}

P2P = {
    "send",
    "recv",
}


class Op:
    """Parses relevant info about operation out of 'event' dict

    examples of supported `profiling_name`s:
        nccl:broadcast
        nccl:send 1->2
        nccl:recv 3<-0
    """

    def __init__(self, event):
        profiling_name = event["profiling_name"]
        nccl, name = profiling_name.split(":")
        assert nccl == "nccl", f"name formatting error? {nccl} != 'nccl'"
        parts = name.split(" ")
        type = parts[0]
        meta = parts[1] if len(parts) == 2 else None

        self.pg_name, _ = event["process_group"]

        assert type in COLLECTIVES | P2P | {
            "coalesced"
        }, f"{type} is not a supported operation"
        self.type = type
        if type == "send":
            s, d = meta.split("->")
            self._src, self._dst = int(s), int(d)
        elif type == "recv":
            d, s = meta.split("<-")
            self._dst, self._src = int(d), int(s)
        else:
            self._src, self._dst = None, None

        if type in P2P | COLLECTIVES:
            self.input_sizes = event["input_sizes"]
            self.output_sizes = event["output_sizes"]
        else:
            self.input_sizes, self.output_sizes = None, None
        self.seq_id = event["seq_id"]

    @property
    def src(self):
        assert self.type in P2P, "can't get src of non-p2p op"
        return self._src

    @property
    def dst(self):
        assert self.type in P2P, "can't get dst of non-p2p op"
        return self._dst

    def __repr__(self):
        if self.type in P2P:
            return f"{self.type}(src={self.src}, dst={self.dst}, input_sizes={self.input_sizes})"
        return f"{self.type}(input_sizes={self.input_sizes})"

    def match(self, other):
        # TODO: I think this can validly not match,
        # e.g. if one PG was used for p2p ops between only some of the peers?
        # if self.seq_id != other.seq_id:
        # return False

        if self.type == "send":
            return (
                other.type == "recv"
                and self.src == other.src
                and self.dst == other.dst
                and self.input_sizes == other.output_sizes
            )
        elif self.type == "recv":
            return (
                other.type == "send"
                and self.src == other.src
                and self.dst == other.dst
                and self.output_sizes == other.input_sizes
            )
        elif self.type in COLLECTIVES:
            return self.type == other.type and self.input_sizes == other.input_sizes
            # TODO(whc) - output sizes dont have to match for e.g. gather, not sure if they ever have to match?
            # and self.output_sizes == other.output_sizes)
        elif self.type == "coalesced":
            return other.type == "coalesced"


def match_one_event(event_a, event_b):
    op_a = Op(event_a)
    op_b = Op(event_b)
    return op_a.match(op_b)


def match_coalesced_groups(all_rank_events, group_size, groups, memberships):
    """
    all_rank_events: {
        rank: [
            (idx, event_dict)
        ]
    }

    Note: it is possible for event dicts in a coalesced group to be asymmetric.
        e.g. the following events lists form a valid coalescing group
             events0 [send:1]
             events1 [recv:0, send:2]
             events2 [recv:1]

    Rule 1: all ops should find a match
    Rule 2: relative ordering of sends and recvs in one event list can be arbitrary
        e.g.
        events1 [recv:0, send:2]  —> okay
        events1 [send:2, recv:0] —> also okay
    Rule 3: sends to the same dest or recvs from the src should be in a consistent order
        e.g.
        rank0 [send:1 (100B), send:1 (1000B)]
        rank1 [recv:0 (1000B), recv:0 (100B)]   —> not okay
    """
    assert (
        len(all_rank_events) == group_size
    ), f"must provide one events list per rank in the group of size {group_size}"
    all_ops = {
        rank: [Op(e) for i, e in all_rank_events[rank]] for rank in all_rank_events
    }

    i = 0
    row = []
    progress = True
    table = []
    while progress:
        progress = False
        for r in all_ops:
            if len(all_ops[r]) > i:
                _, event = all_rank_events[r][i]
                # row.append((event['profiling_name'], event['seq_id'], event['pg_id']))
                row.append(Op(event))
                progress = True
            else:
                row.append(None)
        table.append(row)
        row = []
        i += 1

    print(
        "Ops in coalesce group\n",
        tabulate(table, headers=[f"Rank {i}" for i in range(group_size)]),
    )
    # TODO can't verify seq_id bc there might have been valid seq deltas between ranks even within a pg.
    for op_list in all_ops.values():
        if not op_list:
            print(
                "TODO- not sure if its valid for only some ranks in a PG to participate in a coalesced op?"
            )
            return False
        assert op_list[-1].type == "coalesced"
        op_list.pop(-1)

    while all_ops:
        first_rank = next(iter(all_ops))
        my_ops = all_ops[first_rank]

        if len(all_ops[first_rank]) == 0:
            all_ops.pop(first_rank)
            continue

        # lets match the first collective! we need to know which ranks are involved, and ensure that this same
        # collective is also the first one on those ranks within that group
        op = my_ops[0]
        match_idx = -1
        if op.type in P2P:
            dst_global_rank = sorted(list(memberships[op.pg_name]))[op.dst]
            peer_ops = all_ops[dst_global_rank]
            for i, other in enumerate(peer_ops):
                if op.match(other):
                    match_idx = i
                    break
                elif op.dst == other.src:
                    # Rule 3
                    break
                else:
                    # Rule 1
                    continue
        else:
            raise NotImplementedError("coalesced collective ops")
        if match_idx >= 0:
            my_ops.pop(0)
            peer_ops.pop(match_idx)
        else:
            return False

    return True


"""
Schema for flat DB

TODO schemas not yet implemented
# threads as recorded at termination of process
Threads
	id: int
	traceback_id: int
	process_id: int

Process:
	id: int # Same as world groups RANK
	pid: int
	hostname: str

NCCLOp:
	# nccl op implementation details (sends/recv)
	id: int
	nccl_call_id: int

"""


class Group(NamedTuple):
    id: int
    desc: str
    size: int


class Membership(NamedTuple):
    group_id: Ref[Group]
    global_rank: int


class Traceback(NamedTuple):
    id: int
    frames: str


class Collective(NamedTuple):
    id: int
    group_id: Ref[Group]


class NCCLCall(NamedTuple):
    id: int
    collective_id: Ref[Collective]
    group_id: Ref[Group]
    global_rank: int  # technically Ref[Process] once we have it
    traceback_id: Ref[Traceback]
    collective_type: str
    sizes: List[List[int]]


class Database(NamedTuple):
    groups: List[Group]
    memberships: List[Membership]
    tracebacks: List[Traceback]
    collectives: List[Collective]
    ncclcalls: List[NCCLCall]


types = [
    TypeInfo.from_type(t)
    for t in globals().values()
    if (
        isinstance(t, type)
        and issubclass(t, tuple)
        and hasattr(t, "_fields")
        and t is not TypeInfo
    )
]


def build_nccl_call(entry, id, collective_id, group_id, global_rank):
    return NCCLCall(
        id=id,
        collective_id=collective_id,
        group_id=group_id,
        global_rank=global_rank,
        traceback_id=0,  # TODO
        collective_type=entry["profiling_name"],
        sizes=entry["input_sizes"],
    )


"""
Flat DB builder
"""


def build_groups_memberships(pg_config):
    """
    pg_config: {
        global_rank: {
            (pg_id, desc, ranks)
        }
    }

    `pg_id` is a system generated id, but depending on the mode of PG creation it could be a globally incrementing int
          or a hash of the ranks.  See `_process_group_name` in distributed_c10d.py.
    `desc` is provided by the user (optionally) and should be 'meaningful' (e.g. TP/PP/DP group)
    `ranks` is a list of the 'global ranks' that are members of the PG.

    (pg_id, desc, ranks) tuples are appended lazily to the flight buffer when `getNCCLComm` is called on a PG and
    the `enabled_` flag is true for that PG.
        - the order of calling (init_process_group, new_group, etc) does not affect the order of the tuples in the list

    Returns: a groups table and a membership table, where each row is a Group or Membership namedtuple
    """
    # flat lists for return
    groups = []
    memberships = []

    # dicts for faster cross-rank validation
    _groups = {}
    _memberships = {}
    for global_rank in pg_config:
        for pg_id in pg_config[global_rank]:
            desc = pg_config[global_rank][pg_id]["desc"]
            ranks = pg_config[global_rank][pg_id]["ranks"]
            if isinstance(ranks, str):
                # TODO Bug in FR data format? ranks is '[0, 1,...]'
                ranks = eval(ranks)

            if pg_id not in _groups:
                groups.append(Group(id=pg_id, desc=desc, size=len(ranks)))
                for rank in ranks:
                    memberships.append(Membership(group_id=pg_id, global_rank=rank))
                _groups[pg_id] = groups[-1]
                _memberships[pg_id] = set(ranks)
            else:
                # validation across ranks
                assert (
                    _groups[pg_id].desc == desc
                ), f"mismatch in desc {_groups[pg_id].desc} vs {desc}"
                assert _memberships[pg_id] == set(
                    ranks
                ), f"mismatch in membership {_memberships[pg_id]} vs {set(ranks)}"
    return groups, _groups, memberships, _memberships


def build_nccl_call(entry, id, collective_id, group_id, global_rank):
    return NCCLCall(
        id=id,
        collective_id=collective_id,
        group_id=group_id,
        global_rank=global_rank,
        traceback_id=0,  # TODO
        collective_type=entry["profiling_name"],
        sizes=entry["input_sizes"],
    )


def find_coalesced_group(pg_name, entries):
    """Given a list of entries, if the seq_id of the first entry matches that of subsequent ones,
    build an return a list of entries terminating in a 'coalesed' op entry all sharing a seq_id
    """
    found = []
    seq_id = None
    for i, e in enumerate(entries):
        if e["process_group"][0] != pg_name:
            continue
        elif seq_id is None:
            seq_id = e["seq_id"]
            found.append((i, e))
        elif e["seq_id"] == seq_id:
            found.append((i, e))
        else:
            break

    if len(found) > 1:
        assert found[-1][1]["profiling_name"] == "nccl:coalesced"
        return found
    return []


def build_collectives(all_entries, _groups, _memberships):
    """
    groups, memberships are the non-flat dicts that are indexable
    all_entries is a raw dict from the original dumps:

    all_entries: {
        global_rank: [
            {
                record_id: ordered id of the event in the trace buffer
                pg_id: ProcessGroupNCCL::uid_
                    *note: `pg_id` corresponds to nothing in groups table
                process_group: (pg_name, desc)
                    *note: `pg_name`, `desc` corresponds to `pg_id`, `desc` in groups table
                seq_id: ordered id for collective operations and coalesced group operations
                op_id: ordered id including individual ops inside coalescing group
                profiling_name: descriptive name of the operation
                'time_created_ns',
                'input_sizes',
                'output_sizes',
                'state',
                'time_discovered_started_ns',
                'time_discovered_completed_ns',
                'retired',
                'frames',
            }
        ]
    }
    """
    tracebacks = []
    _tracebacks = {}

    collectives = []
    nccl_calls = []

    # once we find one mismatch, we stop pairing up collectives since the pairing is possibly incorrect
    # instead, just record the remaining ops as NCCLCalls
    mismatch = {_groups[g].id: False for g in _groups}
    """
    - it doesn't matter what order I put collectives/ncclops into their table. we can later on re-sort it by start time
    - there could be multiple options for the "first" collective to pair up (rank 0,1 might do a bcast while rank 2,3 do a bcast)
    - within a group, the first collective must be the same on all ranks in the group, then it can be marked as a collective and removed
    - iterate
    """
    while all_entries:
        # we greedily match collectives, starting arbitrarily with the trace from the first rank
        # later, if we exhaust the first rank, we continue with the next 'first rank'
        rank_iter = iter(all_entries)
        first_rank = next(rank_iter)
        other_ranks = list(rank_iter)

        if len(all_entries[first_rank]) == 0:
            all_entries.pop(first_rank)
            continue

        # lets match the first collective! we need to know which ranks are involved, and ensure that this same
        # collective is also the first one on those ranks within that group
        entries = all_entries[first_rank]
        pg_name, desc = entries[0]["process_group"]
        profiling_name = entries[0]["profiling_name"]
        expected_ranks = set(_memberships[pg_name])
        found_ranks = set([first_rank])
        found_idx = {}

        if find_coalesced_group(pg_name, entries):
            all_coalesced_entries = {
                r: find_coalesced_group(pg_name, all_entries[r])
                if r in all_entries
                else []
                for r in expected_ranks
            }
            match = match_coalesced_groups(
                all_coalesced_entries,
                group_size=_groups[pg_name].size,
                groups=_groups,
                memberships=_memberships,
            )

            if match and not mismatch[pg_name]:
                print("TODO implement the matching path")
                collectives.append(Collective(id=len(collectives), group_id=pg_name))
            else:
                print("Coalesced group mismatch")
                mismatch[pg_name] = True

            for r in all_coalesced_entries:
                reversed_calls = []
                for i, _ in reversed(all_coalesced_entries[r]):
                    reversed_calls.append(
                        build_nccl_call(
                            all_entries[r].pop(i),
                            id=len(nccl_calls),
                            collective_id=collectives[-1].id if match else None,
                            group_id=pg_name,
                            global_rank=r,
                        )
                    )
                nccl_calls.extend(reversed(reversed_calls))

        else:
            for o in expected_ranks.intersection(set(other_ranks)):
                for i, e in enumerate(all_entries[o]):
                    # step over ops from other PGs
                    if e["process_group"] == (pg_name, desc):
                        if match_one_event(entries[0], e) and not mismatch[pg_name]:
                            found_ranks.add(o)
                            found_idx[o] = i
                        else:
                            # we found a mismatch. what do we do with that?
                            mismatch[pg_name] = True
                            print(
                                f"Mismatched collective on rank {o} for group {pg_name}:{desc} collective {profiling_name}"
                            )
                        break

            # at this point there are 3 possibilities
            # 1. we found a match on all the ranks that are members of the group
            #  -> we create a Collective and remove the individual entries from their original lists
            if found_ranks == expected_ranks and not mismatch[pg_name]:
                collectives.append(Collective(id=len(collectives), group_id=pg_name))
                for r in found_ranks:
                    i = found_idx[r] if r != first_rank else 0
                    nccl_calls.append(
                        build_nccl_call(
                            all_entries[r].pop(i),
                            id=len(nccl_calls),
                            collective_id=collectives[-1].id,
                            group_id=pg_name,
                            global_rank=r,
                        )
                    )

            # 2. we found a partial match but some ranks are missing
            # 3. we found no match
            #  -> since its not a complete collective, no entry goes into collectives but we still record a nccl call
            #     TODO should there be a way to mark 'mismatches'?
            else:
                print("appending a non-matching collective")
                nccl_calls.append(
                    build_nccl_call(
                        all_entries[first_rank].pop(0),
                        id=len(nccl_calls),
                        collective_id=None,
                        group_id=pg_name,
                        global_rank=r,
                    )
                )

    return tracebacks, collectives, nccl_calls


def check_no_missing_dump_files(entries, memberships):
    all_ranks = set()
    for membership in memberships:
        all_ranks.add(membership.global_rank)
    dumps_ranks = set(entries.keys())
    assert (
        dumps_ranks == all_ranks
    ), f"Missing dump files from ranks {all_ranks - dumps_ranks}"


def check_version(versions):
    for rank, version in versions.items():
        major, minor = map(int, version.split("."))
        assert major == 1, f"Rank {rank} unsupported version {version}"
        assert minor >= 5, f"Rank {rank} unsupported version {version}"


def check_trace_from_beginning(entries):
    for rank in entries:
        first_record_id = entries[rank][0]["record_id"]
        # TODO add more sequence information such that analysis can proceed even without complete buffer

        assert (
            first_record_id == 0
        ), f"Rank {rank} trace does not start at time 0 (first record is {first_record_id}."
        if first_record_id != 0:
            print(
                f"Rank {rank} trace does not start at time 0 (first record is {first_record_id}."
            )
            return False
    return True


def build_db(details):
    # temporary state used for building database
    entries = {}
    pg_config = {}
    version = {}
    for dump in details.values():
        rank = dump["rank"]
        entries[rank] = dump["entries"]
        version[rank] = dump["version"]
        pg_config[rank] = dump["pg_config"]

    check_version(version)
    check_trace_from_beginning(entries)

    # flattened database
    groups, _groups, memberships, _memberships = build_groups_memberships(pg_config)
    print("built groups, memberships")

    check_no_missing_dump_files(entries, memberships)

    tracebacks, collectives, nccl_calls = build_collectives(
        entries, _groups, _memberships
    )
    print("built collectives, nccl_calls")
    print("Groups\n", tabulate(groups, headers=Group._fields))
    print("Memberships\n", tabulate(memberships, headers=Membership._fields))
    print("Collectives\n", tabulate(collectives, headers=Collective._fields))
    print("NCCLCalls\n", tabulate(nccl_calls, headers=NCCLCall._fields))
    db = Database(
        tracebacks=tracebacks,
        collectives=collectives,
        ncclcalls=nccl_calls,
        groups=groups,
        memberships=memberships,
    )
    return db


def read_dump(filename: str):
    try:
        host_name, _, rank = os.path.basename(filename).split("_", 2)
    except ValueError:
        _, rank = os.path.basename(filename).split("_", 1)
        host_name = "host0"
    rank = int(rank)

    with open(filename, "rb") as infile:
        dump = pickle.load(infile)

    entries = dump["entries"]
    version = dump["version"]
    pg_config = dump["pg_config"]

    return {
        "host_name": host_name,
        "rank": rank,
        "entries": entries,
        "version": version,
        "pg_config": pg_config,
    }


def read_dir(folder: str):
    import gc
    import time

    gc.disable()
    details = {}
    t0 = time.time()
    for root, _, files in os.walk(folder):
        for f in files:
            ta = time.time()
            details[f] = read_dump(os.path.join(root, f))
            tb = time.time()
            print(f"read file {f} in {tb - ta}s")
    print(f"loaded {len(files)} files in {tb - t0}s")
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dir", help="Directory with flight recorder dumps")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    details = read_dir(args.dir)
    db = build_db(details)
    if args.output:
        with open(args.output, "wb") as f:
            pickle.dump((types, db), f)


if __name__ == "__main__":
    main()
