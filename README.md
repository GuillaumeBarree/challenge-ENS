# Algo Challenge

Learning biological properties of molecules from their structure by Simulations Plus.
You can find our recorded presentation on Youtube <a href="https://youtu.be/CUA6e2xEMZ4" title="youtube_video">[here]</a> and our report about the different dimension reduction techniques <a href="https://gitlab-student.centralesupelec.fr/2018barreeg/algo-challenge/-/blob/master/algo_data_science.pdf" title="rapport">[here]</a>

## Launch the training

```bash
cd ./src
python3 train.py --path_to_config ./config.yaml
```

## Launch inference on the test set

```bash
cd ./src
python3 inference.py --path_to_config ./config.yaml
```

## Launch model averaging on the test set

```bash
cd ./src
python3 average_inference.py --path_to_config ./config.yaml
```

## Create the documentation

```bash
pip install virtualenv
python3 -m venv challenge_ds
source challenge_ds/bin/activate
pip --version
pip install -r requirements.txt

cd ./docs
sphinx-apidoc -o ./source ../src
make html

cd ./build/html
firefox index.html
```
