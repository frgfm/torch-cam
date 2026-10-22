#!/bin/sh
set -eu
mkdir -p "$1/src" "$1/.streamlit"
cp demo/app.py "$1/src/streamlit_app.py"
cp demo/inference.py "$1/src/inference.py"
cp demo/Dockerfile "$1/Dockerfile"
cp .streamlit/config.toml "$1/.streamlit/config.toml"
cp demo/requirements.txt "$1/requirements.txt"
printf 'torchcam[demo] @ git+https://github.com/%s.git@%s\n' \
    "$GITHUB_REPOSITORY" "$GITHUB_SHA" >>"$1/requirements.txt"
