
import torch
from torch.distributed.pipelining.schedules import _Action, PipelineScheduleMulti

class SchedulePOPS(PipelineScheduleMulti):
    """
    base schedules for POPs
    """
    def __init__(self) -> None:
        self.num_ranks = 8
        self. num_microbatches = 16
        self.schedule_str = """  """
        self.pipeline_order = []
    
    def _load_str(self, str, format="compute_only")->None:
            """Load a CSV representation of the schedule from a file with the provided filename.
  
            format must be "compute_only" for PipelineScheduleMulti
            """
            assert format == "compute_only"
            rank = 0
            pipeline_ops = {}
            sched_split = self.schedule_str.split("\n")
            for rank_ops in sched_split:
                row = rank_ops.split("[")[1].strip("]")
                pipeline_ops[rank] = row
                print("\n")
                rank += 1
                self.pipeline_order[rank] = [_Action.from_str(s.strip(" ").strip("'")) for s in row.split(",")]

schedule_4rank = "rank 0: ['0F0', '0F1', '0F2', '0F3', '0F4', '0F5', '4F0', '4F1', '4F2', '4F3', '4F4', '4F5', '4B0', '4W0', '4B1', '4W1', '4B2', '4W2', '0B0', '4B3', '0B1', '0W0', '0B2', '4B4', '0W1', '0B3', '4B5', '0W2', '0W3', '0B4', '0W4', '4W3', '0B5', '0W5', '4W4', '4W5']
rank 1: ['1F0', '1F1', '1F2', '1F3', '1F4', '1F5', '5F0', '5F1', '5F2', '5F3', '5F4', '5F5', '5B0', '5W0', '5B1', '5W1', '5B2', '5W2', '1B0', '5B3', '1B1', '1W0', '1B2', '5B4', '1W1', '1B3', '5B5', '1W2', '1W3', '1B4', '1W4', '5W3', '1B5', '1W5', '5W4', '5W5']
rank 2: ['2F0', '2F1', '2F2', '2F3', '2F4', '2F5', '6F0', '6F1', '6F2', '6F3', '6B0', '6F4', '6B1', '6F5', '6B2', '6W0', '2B0', '6B3', '2B1', '2W0', '2B2', '6B4', '2W1', '2B3', '6B5', '2W2', '2W3', '2B4', '2W4', '6W1', '2B5', '2W5', '6W2', '6W3', '6W4', '6W5']
rank 3: ['3F0', '3F1', '3F2', '3F3', '3F4', '3F5', '7F0', '7B0', '7F1', '7B1', '7F2', '7B2', '7F3', '3B0', '7B3', '3B1', '7F4', '3B2', '7B4', '7F5', '3B3', '7B5', '3W0', '3W1', '3B4', '3W2', '3W3', '3B5', '3W4', '3W5', '7W0', '7W1', '7W2', '7W3', '7W4', '7W5']"

schedule_8rank = "rank 0: ['0F0', '0F1', '0F2', '0F3', '0F4', '0F5', '0F6', '0F7', '0F8', '0F9', '0B0', '0W0', '0B1', '0W1', '0B2', '0W2', '0B3', '0W3', '0B4', '0W4', '0B5', '0W5', '0B6', '0W6', '0B7', '0W7', '0B8', '0W8', '0B9', '0W9']
rank 1: ['1F0', '1F1', '1F2', '1F3', '1F4', '1F5', '1F6', '1F7', '1F8', '1F9', '1B0', '1W0', '1B1', '1W1', '1B2', '1W2', '1B3', '1W3', '1B4', '1W4', '1B5', '1W5', '1B6', '1W6', '1B7', '1W7', '1B8', '1W8', '1B9', '1W9']
rank 2: ['2F0', '2F1', '2F2', '2F3', '2F4', '2F5', '2F6', '2F7', '2F8', '2F9', '2B0', '2W0', '2B1', '2W1', '2B2', '2W2', '2B3', '2W3', '2B4', '2W4', '2B5', '2W5', '2B6', '2W6', '2B7', '2W7', '2B8', '2W8', '2B9', '2W9']
rank 3: ['3F0', '3F1', '3F2', '3F3', '3F4', '3F5', '3F6', '3F7', '3F8', '3F9', '3B0', '3W0', '3B1', '3W1', '3B2', '3W2', '3B3', '3W3', '3B4', '3W4', '3B5', '3W5', '3B6', '3W6', '3B7', '3W7', '3B8', '3W8', '3B9', '3W9']
rank 4: ['4F0', '4F1', '4F2', '4F3', '4F4', '4F5', '4F6', '4F7', '4F8', '4F9', '4B0', '4W0', '4B1', '4W1', '4B2', '4W2', '4B3', '4W3', '4B4', '4W4', '4B5', '4W5', '4B6', '4W6', '4B7', '4W7', '4B8', '4W8', '4B9', '4W9']
rank 5: ['5F0', '5F1', '5F2', '5F3', '5F4', '5F5', '5F6', '5B0', '5F7', '5B1', '5F8', '5B2', '5F9', '5B3', '5W0', '5B4', '5W1', '5B5', '5W2', '5B6', '5W3', '5B7', '5W4', '5B8', '5W5', '5B9', '5W6', '5W7', '5W8', '5W9']
rank 6: ['6F0', '6F1', '6F2', '6F3', '6B0', '6F4', '6B1', '6F5', '6B2', '6F6', '6B3', '6F7', '6B4', '6F8', '6B5', '6F9', '6B6', '6W0', '6B7', '6W1', '6B8', '6W2', '6B9', '6W3', '6W4', '6W5', '6W6', '6W7', '6W8', '6W9']
rank 7: ['7F0', '7B0', '7F1', '7B1', '7F2', '7B2', '7F3', '7B3', '7F4', '7B4', '7F5', '7B5', '7F6', '7B6', '7F7', '7B7', '7F8', '7B8', '7F9', '7B9', '7W0', '7W1', '7W2', '7W3', '7W4', '7W5', '7W6', '7W7', '7W8', '7W9']"
