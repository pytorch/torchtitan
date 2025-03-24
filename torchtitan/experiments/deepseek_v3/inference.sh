
#!/bin/bash
# Get the prompt from command line argument or use a default
prompt="${1:-What is 2+2?}"

# Run the model with the prompt
torchrun --standalone --nproc-per-node 4 generate.py "$prompt"
