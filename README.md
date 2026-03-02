# Geuse

Hand rehabilitation prototype using MediaPipe hand landmarks + multi-task neural network (classification + closure regression).

## Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

## Scripts
- build_dataset.py
- train_multiclass_nn.py
- train_multitask.py
- realtime_multitask_demo.py

## Dataset used
[HaGRID](https://github.com/hukenovs/hagrid)