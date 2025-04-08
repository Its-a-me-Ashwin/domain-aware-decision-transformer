sudo docker run --gpus all -it --rm   -v $(pwd):/workspace   -w /workspace   jax-pytorch-gpu   python env.py 
