# Project context
- For the project context and instructions, use the same as Claude Code from the `CLAUDE.md` in the project root dir
- Some core libraries in this project were implemented referring to the `Diffusers` project, which
  - is accessible locally at `~/workspace/custom-builds/diffusers` 
  - is hosted at `https://github.com/huggingface/diffusers`
  - refer to it when analyzing the core library implementations in this project.
- This project uses the base model `Tongyi-MAI/Z-Image-Turbo` by default; read the `docs/z-image-turbo.md` to understand the model structure.

# Useful resources
- For analyzing `.safetensors` file structure, you can use the python script `~/bin/stls.py`
  - use `--format toon` to ouput in the LLM friendly format "TOON"
  - if the script is not present, it's downloadable via `curl https://gist.githubusercontent.com/zhutao100/cc481d2cd248aa8769e1abb3887facc8/raw/89d644c490bcf5386cb81ebcc36c92471f578c60/stls.py > ~/bin/stls.py`
- To inpsect the base model used by the project by default, read from the latest snapshot at `~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots`
