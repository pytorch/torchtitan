# Q3: Data Loading & Sharding

Trace the initialization of the dataset and dataloader. Identify the file, class,
and line number where the global dataset is sliced or partitioned among the Data
Parallel ranks to ensure each rank receives a unique, non-overlapping subset of
data.
