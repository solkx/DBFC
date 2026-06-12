# DBFC

### 1. Prepare dataset

Download the Twitter-15 and Twitter-17 datasets [here](https://drive.google.com/file/d/1ogfbn-XEYtk9GpUECq1-IwzINnhKGJqy/view)

### 2. Generate images based on mllm

Download MLLM from [here](https://huggingface.co/zai-org/CogView4-6B), and generate images by run `python gen_image.py`.

### 3. Train BFC-MMGM

First, enter the `mmgm` directory, then train BFC-MMGM by run `python main.py`.

### 4. Train downstream MNER model

First, enter the `mner` directory, then train MNER model by run `python run.py`.

