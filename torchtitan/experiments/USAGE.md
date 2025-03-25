## Running DeepSeek in Titan  (experimenatal)

1 - You will need to download a DeepSeek model weights if you want to run a pre-trained checkpoint.
You can simply run "python download.py [vX]  where vX = v2 or v3, both are supported.

2 - Running inference:
You can run inference by using "bash inference.sh" which will run v2 by default using 4 gpus.
A default prompt is provided but you can modify this.
