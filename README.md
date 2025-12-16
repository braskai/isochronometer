# [WMT24 @ EMNLP24] IsoChronoMeter (**Official Paper Repo**)

A minimal library for machine translation (MT) quality estimation (qe) based on both classical and duration based predictors.

Here is the paper:
[https://arxiv.org/abs/2410.11127]


---
## Getting Started (Tested 16/12/2025)

- Known important issues: Japanese and Italian are not supported by duration predictor! (Need to add alternatives.)


---
### 0. Pre-requisite
1. You need the library: `libsndfile` & `openfst`
Depending on your platform (MAC / Linux) you will need different commands. E.g.:

- Linux: (Conda variant tested, libsndfile probably will require apt in non conda)
```bash
conda create -n env_isochronometer python=3.11
conda activate env_isochronometer
conda install -c conda-forge libsndfile==1.0.31
```

- Mac (**now working**): we have a workaround (https://github.com/wenet-e2e/WeTextProcessing/issues/317#issuecomment-3660590983):
```bash
virtualenv env_isochronometer
source env_isochronometer/bin/activate
brew install libsndfile
# brew install openfst
```

---
### 1. Installation

1. Installation on Linux (CONDA)
```bash
pip3 install sonar-space==0.5.0 torch==2.8.0 #needs a seperate install (due to numpy)

# Install lilcom
conda install -c lilcom lilcom #if you use conda, use `pip3 install lilcom==1.1.0` (if pip)

# Install the general requirements
pip3 install -r requirements.txt

# Install WeTextProcessing
pip3 install WeTextProcessing==1.0.4.1

# install isochronometer
pip3 install -e .
```


2. Installation from Source MacOS: (Openfst 1.8.4 & pynini 2.1.7, https://github.com/wenet-e2e/WeTextProcessing/issues/317)
```bash
pip3 install sonar-space==0.5.0 torch==2.8.0 #needs a seperate install (due to numpy)

# Install lilcom
pip3 install lilcom==1.1.0  #if you use pip

# Install the general requirements
pip3 install -r requirements.txt

# Install Pynini
brew install openfst
export CPLUS_INCLUDE_PATH="/opt/homebrew/include:$CPLUS_INCLUDE_PATH"
export LIBRARY_PATH="/opt/homebrew/lib:$LIBRARY_PATH"
pip3 install uv
uv pip install pynini==2.1.7

# Install WeTextProcessing
pip3 install WeTextProcessing==1.0.4.1 --no-deps

pip3 install "numpy<2.0.0" #some libraries require <2 numpy
# install isochronometer
pip3 install -e .
```

---
### 2. Testing the installation

1. Testing Blaser (classical MT Quality Estimation)
```bash
python3 -m isochronometer.test_blaser
```

2. Testing isochrony (duration-based MT Quality Estimation)
```bash
python3 -m isochronometer.test_isochrony
```

---
### 3. Usage:
See `scripts/*` files for more details.

1. Evaluating blaser:
```python
from isochronometer import blaser

# sample input
original_sentences = ["this is a sentence.", "this is another sentence."]   
translated_sentences = ["das ist ein satz.", "das ist ein anderer satz."]

# calculations
model = blaser.get_model()
blaser_score = model.score(original_sentences, translated_sentences, source_lang="en", translated_lang="de")
print(f"Blaser Score: {blaser_score}")
```

2. Evaluating isochrony:
```python
from isochronometer import isochrony

# sample input
original_sentences = ["this is a sentence.", ""]   
translated_sentences = ["das ist ein satz.", ""]
original_start_times = ["00:00:40,025","00:00:46,167"]
original_end_times = ["00:00:46,167","00:00:46,167"]

# calculations
metric = isochrony.get_metric(source_lang="en", translated_lang="de")

score = metric.evaluate(metric_type="ICM", original_sentences=original_sentences, translated_sentences=translated_sentences)
print(f"ICM:{score}")

score = metric.evaluate(metric_type="pred-SO", original_sentences=original_sentences, translated_sentences=translated_sentences)
print(f"pred-SO:{score}")

score = metric.evaluate( #SO requires start and end times.
    metric_type="SO", 
    original_sentences=original_sentences, 
    translated_sentences=translated_sentences, 
    original_start_times=original_start_times, 
    original_end_times=original_end_times
)
print(f"SO:{score}")

```



---

Please cite this work:
```bibtex
@misc{rozanov2024isochronometersimpleeffectiveisochronic,
      title={IsoChronoMeter: A simple and effective isochronic translation evaluation metric}, 
      author={Nikolai Rozanov and Vikentiy Pankov and Dmitrii Mukhutdinov and Dima Vypirailenko},
      year={2024},
      eprint={2410.11127},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.11127}, 
}
```

## (C) 2024 - Present, Nikolai Rozanov et al.


