# Evaluation

## Open-loop Evaluation
Open-loop evaluation validates the model's predictive capabilities under fixed historical conditions.

**Configuration**
1. Update the [config file](/config/eval/DrivePi0/open_loop.yaml)
2. **Important**: The model must be trained with **horizon=20** (predicting the next 20 trajectory points)

**Run Evaluation**
```bash
bash script/evaluation/open_loop_drivepi0.sh
```

## Closed-loop evaluation
For closed-loop testing, configure the [config file](/config/eval/DrivePi0/closed_loop.yaml) (update paths, etc.). You can download our DrivePi0 checkpoint [here](https://huggingface.co/rethinklab/DrivePi0).

The following are our evaluation results on 8*H200 GPUs.

| Model | DS | SR |
|:-------:|:-------:|:-------:|
| [DrivePi0-Base-float16](/docs/drivepi_base_float16.json) | 55.85 | 30.00 |
| [DrivePi0-Base-float32](/docs/drivepi_base_float32.json) | 65.85 | 42.27 |
| [DrivePi0-Full-float32](/docs/drivepi_full_float32.json) | 67.41 | 44.09 |

To properly set up the evaluation environment, please clone the [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) repository at the same directory level as your DriveMoE repository
```console
cd ..
git clone https://github.com/Thinklab-SJTU/Bench2Drive.git
```
This will result in the following directory structure:
```
/parent_directory/
├── DriveMoE/
│   ├── ... (DriveMoE contents)
└── Bench2Drive/
    ├── ... (Bench2Drive contents)
```

Download and setup CARLA 0.9.15, you can skip if you have already downloaded CARLA on your device.
```console
mkdir carla
cd carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xvf CARLA_0.9.15.tar.gz
cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd .. && bash ImportAssets.sh
```

Then link team_code like this:
```console
cp DriveMoE/script/evaluation/closed_loop_drivepi0.sh Bench2Drive/leaderboard/scripts
cp DriveMoE/script/evaluation/requirements.txt Bench2Drive
mkdir Bench2Drive/leaderboard/team_code
cd Bench2Drive/leaderboard/team_code
ln -s ../../../DriveMoE/src/agent/team_code/*  ./
cd ../../   # Now you are in the Bench2Drive directory
```

Then set up eval environment
```console
conda create -n DriveMoE_eval python=3.8
conda activate DriveMoE_eval
export CARLA_ROOT=YOUR_CARLA_PATH
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> YOUR_CONDA_PATH/envs/DriveMoE_eval/lib/python3.8/site-packages/carla.pth
pip install -r requirements.txt
```
Set closed_loop_drivepi0.sh basing on your device and then run:

```bash
bash leaderboard/scripts/closed_loop_drivepi0.sh
```

Follow [this](https://github.com/Thinklab-SJTU/Bench2Drive?tab=readme-ov-file#eval-tools) to use evaluation tools of Bench2Drive.

### Further Assistance
For questions regarding:
- Benchmark implementation details
- Dataset specifications
- Evaluation metrics

Please refer to the [Bench2Drive Documentation](https://github.com/Thinklab-SJTU/Bench2Drive) or open an issue in the [Bench2Drive Issues](https://github.com/Thinklab-SJTU/Bench2Drive/issues) tracker.