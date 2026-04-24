# Data Consolidation

This repository combines our efforts to consolidate datasets for common training
and benchmarking.

# Setup

To run the code on the **JSC supercomputer**, first create the virtual environment
(including loading the git submodules) via:

```bash
source setup/setup.sh
```

In subsequent terminals, the environment can be activated with:

```bash
source setup/activate.sh
```

If you are running this repository on a system **without** the `module` software
and corresponding JSC prebuilt packages, the pip requirements can
be found in `setup/requirements_list.txt`.  
This file is generated using `pip list` (not `pip freeze`), because the JSC
modules point to locally built packages that do not correspond to reproducible
online versions.  
In such a case, also ensure that **FFmpeg 7.0.2 or newer** is available on your system
via the command line.

# Basic Benchmarking

To run the benchmarking, execute:

```bash
python src/benchmarkingdatasets.py
```

Example results are available in:

- `results/sample_datasets.txt`
- `results/benchmarking_video_qt28_veryslow.mp4` (sample camera image video)

Below is an example of the dataset speed comparison, measured on the
**JUWELS login nodes** (with up to 20 CPU cores available).  
*Note that processes may benefit from idle cores during benchmarking; therefore,
the wall time may be significantly lower than the cumulative CPU process time.*

<!-- Replace: \s*-\s+(\w+):\s+([\d.]+)\s+s/it\s+\(\s*([\d.]+)\s+s\s+walltime/it,\s+([\d.]+)\s+s/it/frame\)\s+for\s+(\d+)\s+frames
     with: | $1 |   $2 |           $3 |      $4 |      $5 |
   -->
| Dataset      |   s/it  | walltime/it (s) | s/it/frame | Frames |
|--------------|---------|-----------------|------------|--------|
| aumovio      |   0.892 |           0.171 |      0.892 |      1 |
| avl          |   0.507 |           0.131 |      0.254 |      2 |
| denso        |   0.660 |           0.186 |      0.165 |      4 |
| dspace       |   0.812 |           0.112 |      0.812 |      1 |
| ipg          |   1.858 |           0.300 |      1.858 |      1 |
| fzi          |  13.079 |           1.929 |      2.180 |      6 |
| mb_tld       |   0.652 |           0.160 |      0.163 |      4 |
| mb_ecp       |   0.858 |           0.265 |      0.107 |      8 |
| mb_aw_videos |  36.148 |           3.317 |      0.723 |     50 |
| valeo        |   1.922 |           0.252 |      1.922 |      1 |

If you have comments on any particular dataset, either leave them in `NOTES.md`
for discussion or contact the respective partner directly.

# Adding Your Dataset to the Benchmark

To add a dataset to this benchmarking framework:

1. **Add your Python implementation** under `src/[your_institute_name]/`

2. **Add a configuration entry** describing how to instantiate your dataset in
   `config/sample_datasets.yaml` as demonstrated by the existing examples.
   Class instantiation is handled by the function `instantiate_from_config` in
   `src/utils.py`, should you need additional details.

3. **Ensure your dataset files** referenced in the config are located under
   `/p/data1/nxtaim/proprietary_samples/`, are GDPR-compliant (e.g. no recognizable
   faces or license plates), and that all users have read access to the files.
   A minimal dataset is sufficient for the sample benchmark.

4. Check `NOTES.md` for any particularities of your dataset that other partners may
   have noticed and that need addressing.

# Further Goals

We aim to extend the benchmarking in two directions:

## **1. Multi-worker / GPU-parallel benchmark**
A large dataset run where each dataset is loaded by multiple workers (at least 6)
to efficiently utilize the 12 GPUs available per GPU node.  
Note: this *reduces* the number of open files and available RAM per dataset,
since each worker copies a clone of the dataset structure. Hence, the code may need to
be optimised for these constraints.

## **2. Full large-scale data benchmark**
A benchmark using each partner’s *real* large-scale data.  
This may need to be run by the partner themselves or by the JSC team, depending
on access permissions.  
To support this, partners must ensure that either:
- the directory structure under `/p/data1/nxtaim/proprietary/` matches that of
  the sample dataset, **or**
- an additional configuration file is provided describing how to instantiate the
  dataset on the large-scale data.
