# Debug Model

## Upstream

```
[rank0]:[titan] 2025-10-30 14:32:58,897 - root - INFO - step:  1  loss:  8.1661  grad_norm:  3.6855  memory:  2.47GiB(3.12%)  tps: 3,502  tflops: 1.90  mfu: 0.19%
[rank0]:[titan] 2025-10-30 14:32:58,897 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[rank0]:[titan] 2025-10-30 14:32:58,990 - root - INFO - step:  2  loss:  6.1098  grad_norm:  4.6416  memory:  2.49GiB(3.14%)  tps: 177,138  tflops: 96.17  mfu: 9.72%
[rank0]:[titan] 2025-10-30 14:32:59,077 - root - INFO - step:  3  loss:  4.7947  grad_norm:  2.4418  memory:  2.49GiB(3.14%)  tps: 189,694  tflops: 102.99  mfu: 10.41%
[rank0]:[titan] 2025-10-30 14:32:59,163 - root - INFO - step:  4  loss:  4.5908  grad_norm:  2.4407  memory:  2.49GiB(3.14%)  tps: 189,572  tflops: 102.93  mfu: 10.41%
[rank0]:[titan] 2025-10-30 14:32:59,251 - root - INFO - step:  5  loss:  4.3347  grad_norm:  2.0462  memory:  2.49GiB(3.14%)  tps: 187,152  tflops: 101.61  mfu: 10.27%
[rank0]:[titan] 2025-10-30 14:32:59,342 - root - INFO - step:  6  loss:  4.1779  grad_norm:  1.9530  memory:  2.49GiB(3.14%)  tps: 180,490  tflops: 97.99  mfu: 9.91%
[rank0]:[titan] 2025-10-30 14:32:59,431 - root - INFO - step:  7  loss:  4.0245  grad_norm:  1.7682  memory:  2.49GiB(3.14%)  tps: 184,693  tflops: 100.28  mfu: 10.14%
[rank0]:[titan] 2025-10-30 14:32:59,517 - root - INFO - step:  8  loss:  3.9374  grad_norm:  1.6348  memory:  2.49GiB(3.14%)  tps: 192,383  tflops: 104.45  mfu: 10.56%
[rank0]:[titan] 2025-10-30 14:32:59,603 - root - INFO - step:  9  loss:  3.9923  grad_norm:  1.4602  memory:  2.49GiB(3.14%)  tps: 191,149  tflops: 103.78  mfu: 10.49%
[rank0]:[titan] 2025-10-30 14:32:59,693 - root - INFO - step: 10  loss:  3.8790  grad_norm:  1.4802  memory:  2.49GiB(3.14%)  tps: 183,058  tflops: 99.39  mfu: 10.05%
```

## This Branch

```
[rank0]:[titan] 2025-10-30 14:57:42,612 - root - INFO - step:  1  loss:  8.2308  grad_norm:  3.5201  memory:  2.53GiB(3.20%)  tps: 4,355  tflops: 2.36  mfu: 0.24%
[rank0]:[titan] 2025-10-30 14:57:42,612 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[rank0]:[titan] 2025-10-30 14:57:42,705 - root - INFO - step:  2  loss:  6.2584  grad_norm:  4.6478  memory:  2.55GiB(3.22%)  tps: 176,131  tflops: 95.63  mfu: 9.67%
[rank0]:[titan] 2025-10-30 14:57:42,792 - root - INFO - step:  3  loss:  4.8833  grad_norm:  3.1053  memory:  2.55GiB(3.22%)  tps: 188,610  tflops: 102.40  mfu: 10.35%
[rank0]:[titan] 2025-10-30 14:57:42,881 - root - INFO - step:  4  loss:  4.7526  grad_norm:  2.8135  memory:  2.55GiB(3.22%)  tps: 186,254  tflops: 101.12  mfu: 10.22%
[rank0]:[titan] 2025-10-30 14:57:42,971 - root - INFO - step:  5  loss:  4.4948  grad_norm:  2.4701  memory:  2.55GiB(3.22%)  tps: 182,708  tflops: 99.20  mfu: 10.03%
[rank0]:[titan] 2025-10-30 14:57:43,063 - root - INFO - step:  6  loss:  4.2378  grad_norm:  2.0343  memory:  2.55GiB(3.22%)  tps: 178,396  tflops: 96.86  mfu: 9.79%
[rank0]:[titan] 2025-10-30 14:57:43,156 - root - INFO - step:  7  loss:  4.1080  grad_norm:  2.0472  memory:  2.55GiB(3.22%)  tps: 176,565  tflops: 95.86  mfu: 9.69%
[rank0]:[titan] 2025-10-30 14:57:43,243 - root - INFO - step:  8  loss:  4.0309  grad_norm:  2.0197  memory:  2.55GiB(3.22%)  tps: 189,521  tflops: 102.90  mfu: 10.40%
[rank0]:[titan] 2025-10-30 14:57:43,330 - root - INFO - step:  9  loss:  4.0710  grad_norm:  1.6753  memory:  2.55GiB(3.22%)  tps: 187,713  tflops: 101.92  mfu: 10.30%
[rank0]:[titan] 2025-10-30 14:57:43,422 - root - INFO - step: 10  loss:  3.9525  grad_norm:  1.6444  memory:  2.55GiB(3.22%)  tps: 179,513  tflops: 97.46  mfu: 9.85%
```

# 16B

## Upstream

```
[rank0]:[titan] 2025-10-30 20:34:24,804 - root - INFO - step:  1  loss: 12.0388  grad_norm:  1.7301  memory: 37.96GiB(47.94%)  tps: 1,689  tflops: 30.58  mfu: 3.09%
[rank0]:[titan] 2025-10-30 20:34:24,804 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[rank0]:[titan] 2025-10-30 20:34:38,183 - root - INFO - step: 10  loss: 11.1385  grad_norm:  2.4909  memory: 52.53GiB(66.33%)  tps: 11,023  tflops: 199.58  mfu: 20.18%
[rank0]:[titan] 2025-10-30 20:34:52,342 - root - INFO - step: 20  loss:  9.3949  grad_norm:  4.6760  memory: 52.53GiB(66.33%)  tps: 11,572  tflops: 209.52  mfu: 21.19%
[rank0]:[titan] 2025-10-30 20:35:06,466 - root - INFO - step: 30  loss:  8.4717  grad_norm:  2.9549  memory: 52.53GiB(66.33%)  tps: 11,601  tflops: 210.05  mfu: 21.24%
[rank0]:[titan] 2025-10-30 20:35:20,681 - root - INFO - step: 40  loss:  7.7499  grad_norm:  1.2642  memory: 52.53GiB(66.33%)  tps: 11,527  tflops: 208.71  mfu: 21.10%
[rank0]:[titan] 2025-10-30 20:35:33,463 - root - INFO - [GC] Performing periodic GC collection took 0.09 seconds
[rank0]:[titan] 2025-10-30 20:35:34,953 - root - INFO - step: 50  loss:  7.1839  grad_norm:  0.9971  memory: 52.53GiB(66.33%)  tps: 11,481  tflops: 207.87  mfu: 21.02%
[rank0]:[titan] 2025-10-30 20:35:49,197 - root - INFO - step: 60  loss:  6.8562  grad_norm:  0.9029  memory: 52.53GiB(66.33%)  tps: 11,503  tflops: 208.27  mfu: 21.06%
[rank0]:[titan] 2025-10-30 20:36:03,613 - root - INFO - step: 70  loss:  6.7472  grad_norm:  0.7783  memory: 52.53GiB(66.33%)  tps: 11,366  tflops: 205.80  mfu: 20.81%
[rank0]:[titan] 2025-10-30 20:36:18,009 - root - INFO - step: 80  loss:  6.4662  grad_norm:  1.1184  memory: 52.53GiB(66.33%)  tps: 11,382  tflops: 206.09  mfu: 20.84%
[rank0]:[titan] 2025-10-30 20:36:32,253 - root - INFO - step: 90  loss:  6.3386  grad_norm:  1.1675  memory: 52.53GiB(66.33%)  tps: 11,503  tflops: 208.28  mfu: 21.06%
[rank0]:[titan] 2025-10-30 20:36:44,919 - root - INFO - [GC] Performing periodic GC collection took 0.11 seconds
[rank0]:[titan] 2025-10-30 20:36:46,413 - root - INFO - step: 100  loss:  6.4956  grad_norm:  1.8129  memory: 52.53GiB(66.33%)  tps: 11,572  tflops: 209.53  mfu: 21.19%
```

## This Branch

```
[rank0]:[titan] 2025-10-30 20:16:42,705 - root - INFO - step:  1  loss: 12.0097  grad_norm:  1.7039  memory: 38.12GiB(48.14%)  tps: 400  tflops: 7.25  mfu: 0.73%
[rank0]:[titan] 2025-10-30 20:16:42,705 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[rank0]:[titan] 2025-10-30 20:16:55,811 - root - INFO - step: 10  loss: 11.1487  grad_norm:  2.4410  memory: 52.78GiB(66.65%)  tps: 11,251  tflops: 203.71  mfu: 20.60%
[rank0]:[titan] 2025-10-30 20:17:10,111 - root - INFO - step: 20  loss:  9.4713  grad_norm:  5.3761  memory: 52.78GiB(66.65%)  tps: 11,459  tflops: 207.47  mfu: 20.98%
[rank0]:[titan] 2025-10-30 20:17:24,409 - root - INFO - step: 30  loss:  8.5264  grad_norm:  3.8040  memory: 52.78GiB(66.65%)  tps: 11,460  tflops: 207.49  mfu: 20.98%
[rank0]:[titan] 2025-10-30 20:17:38,756 - root - INFO - step: 40  loss:  7.7817  grad_norm:  1.2021  memory: 52.78GiB(66.65%)  tps: 11,421  tflops: 206.78  mfu: 20.91%
[rank0]:[titan] 2025-10-30 20:17:51,691 - root - INFO - [GC] Performing periodic GC collection took 0.10 seconds
[rank0]:[titan] 2025-10-30 20:17:53,202 - root - INFO - step: 50  loss:  7.1972  grad_norm:  1.0293  memory: 52.78GiB(66.65%)  tps: 11,343  tflops: 205.37  mfu: 20.77%
[rank0]:[titan] 2025-10-30 20:18:07,584 - root - INFO - step: 60  loss:  6.8584  grad_norm:  1.0506  memory: 52.78GiB(66.65%)  tps: 11,392  tflops: 206.26  mfu: 20.86%
[rank0]:[titan] 2025-10-30 20:18:22,114 - root - INFO - step: 70  loss:  6.7502  grad_norm:  0.8394  memory: 52.78GiB(66.65%)  tps: 11,276  tflops: 204.17  mfu: 20.64%
[rank0]:[titan] 2025-10-30 20:18:36,621 - root - INFO - step: 80  loss:  6.4666  grad_norm:  1.4534  memory: 52.78GiB(66.65%)  tps: 11,295  tflops: 204.50  mfu: 20.68%
[rank0]:[titan] 2025-10-30 20:18:50,994 - root - INFO - step: 90  loss:  6.3324  grad_norm:  0.7169  memory: 52.78GiB(66.65%)  tps: 11,400  tflops: 206.40  mfu: 20.87%
[rank0]:[titan] 2025-10-30 20:19:03,791 - root - INFO - [GC] Performing periodic GC collection took 0.09 seconds
[rank0]:[titan] 2025-10-30 20:19:05,301 - root - INFO - step: 100  loss:  6.4607  grad_norm:  0.8626  memory: 52.78GiB(66.65%)  tps: 11,452  tflops: 207.35  mfu: 20.97%
[rank0]:[titan] 2025-10-30 20:19:19,762 - root - INFO - step: 110  loss:  6.2957  grad_norm:  0.7562  memory: 52.78GiB(66.65%)  tps: 11,330  tflops: 205.15  mfu: 20.74%
[rank0]:[titan] 2025-10-30 20:19:34,362 - root - INFO - step: 120  loss:  6.1344  grad_norm:  0.8170  memory: 52.78GiB(66.65%)  tps: 11,222  tflops: 203.19  mfu: 20.55%
[rank0]:[titan] 2025-10-30 20:19:48,736 - root - INFO - step: 130  loss:  6.0316  grad_norm:  0.9751  memory: 52.78GiB(66.65%)  tps: 11,399  tflops: 206.39  mfu: 20.87%
```
