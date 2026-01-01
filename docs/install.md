## Installation
Before you begin, you need to ensure that your CUDA version is >= 12.1.

Clone this repository at your directory and run `pip install -e .` to install environment.

Download [PaliGemma](https://huggingface.co/blog/paligemma) weights to your directory.
```console
git clone https://huggingface.co/google/paligemma-3b-pt-224
```

If you wish to attempt training Drive $\pi_0$ and DriveMoE using the code, or to try open-loop testing with provided checkpoints, you will need to utilize the [Bench2Drive Dataset](https://huggingface.co/datasets/rethinklab/Bench2Drive) and our Camera Labels. Additionally, to accelerate the training process of DriveMoE, we provide a pre-trained PaliGemma model along with the corresponding camera position embedding parameters required for DriveMoE. You can get the labels and pretrained models [here](https://huggingface.co/rethinklab/DriveMoE).


Set environment variables by running `source scripts/set_path.sh`

Then run the script to preprocess training data.
```bash
bash script/generate_data.sh
```
For OPEN-LOOP evaluation (testing with ground truth history), you must be set HORIZON_SIZE to 20 to ensure fair comparison with baseline methods. For CLOSED-LOOP evaluation (testing with predicted history), it can be set to any value based on your requirements.