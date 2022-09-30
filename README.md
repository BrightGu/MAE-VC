# MAE-VC
Voice Conversion Using Learnable Similarity-Guided Masked Autoencoder
For the audio samples, please refer to our [demo page](https://brightgu.github.io/MAE-VC/). The more converted speeches can be found in "Demo/ConvertedSpeeches/".
### Envs
You can install the dependencies with
```bash
pip install -r requirements.txt
```

### Infer
Please refer to "MaskVC/infer/infer.py", and edit "input file path" and "output dir".
```bash
src_file = r"p364_237.wav"
tar_file = r"p241_352.wav"
out_dir = ""
os.makedirs(out_dir, exist_ok=True)
solver.infer(src_file,tar_file,out_dir)
```
### Train from scratch
####  Preprocessing
Please refer to "MaskVC/predata/robust_mels.py"
```bash
train_wav_dir = r"VCTK-Corpus/wav22050/"
    config_path = r"MaskVC/hifi_config.yaml"
    out_dir = r""
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    get_hifi_mels(train_wav_dir,out_dir,config)
```
#### Training
Please config file "MaskVC/hifi_config.yaml".
```bash
# train corpus, from MaskVC/predata/robust_mels.py
label_clip_mel_pkl: "figure_label_mel_map.pkl"
# used for model evaluation during training. test_dir/*.wav, require 22050hz
test_wav_dir: "test_dir"
# model,log,test output
out_dir: ""

# start training
python MaskVC/solver.py
```
