VENV_PATH="$HOME/uv_env/torchtitan"
uv venv "$VENV_PATH" --python "$(which python3.12)" --seed

source $VENV_PATH/bin/activate
uv pip install -r requirements.txt

