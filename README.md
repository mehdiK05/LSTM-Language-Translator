# LSTM Language Translator

A neural machine translation system built using Long Short-Term Memory (LSTM) networks to translate between languages.

## Description

This project implements a sequence-to-sequence (Seq2Seq) model with LSTM architecture for language translation. The system is designed to translate text from one language to another using deep learning techniques.

The model utilizes:
- Encoder-decoder architecture
- LSTM (Long Short-Term Memory) cells to capture long-range dependencies
- Attention mechanism to improve translation quality
- Word embeddings to represent vocabulary

## Features

- Preprocessing pipeline for text data
- Tokenization and vocabulary building
- Training framework for the translation model
- Inference mechanism for translating new sentences
- Evaluation metrics for translation quality

## Installation

```bash
# Clone the repository
git clone https://github.com/mehdiK05/LSTM-Language-Translator.git
cd LSTM-Language-Translator

```

## Usage

### Data Preparation

The translation model requires parallel text data (sentences in source and target languages). The dataset should be organized with aligned source and target sentences.

### English-Darija Dataset
The core idea of the project is to discover the opportunities and challenges of translating Darija sentences to English. We are using a Darija/English parallel dataset from AtlasIA, applying various cleaning methods detailed in the Data Cleaning.ipynb file to obtain a pure dataset of 555K sentences.
You can download the cleaned, ready-for-training English-Darija dataset from Hugging Face:
https://huggingface.co/datasets/midox05/train_ready_english_darija_dataset/tree/main

To load the dataset:

```python
import pandas as pd
# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_csv("hf://datasets/midox05/train_ready_english_darija_dataset/ready_df.csv")
```

### Training

To train the model:

```bash
!python src/train.py \
    --data_file {data_path} \
    --output_dir ./trained_model \
    --embed_dim 256 \
    --hidden_dim 512 \
    --n_layers 2 \
    --dropout 0.5 \
    --bidirectional \
    --attn general \
    --batch_size 64 \
    --epochs 20 \
    --lr 0.001 \
    --clip 1.0 \
    --patience 5
```

## Model Architecture

The translation system uses a Seq2Seq architecture consisting of:

1. **Encoder**: Processes the input sentence and outputs its representation
   - Embedding layer for word representation
   - Bidirectional LSTM layers to capture context

2. **Attention Mechanism**: Helps the model focus on relevant parts of the source sentence

3. **Decoder**: Generates the translated output one word at a time
   - LSTM layers for sequence generation
   - Dense layer for word prediction

## Performance

The model performance depends on:
- Dataset size and quality
- Training time
- Model hyperparameters
- Language pair complexity

## Future Improvements

- Implement beam search for better translation quality
- Add support for more language pairs
- Optimize for deployment on edge devices
- Incorporate transformer architecture for comparison

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
If you have computational resources, you can train the model on the Darija/English dataset or any other dataset and share the results with us.

## Acknowledgments

- This project was inspired by various research papers on neural machine translation mostly from : 
[Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention based neural machine translation.](https://arxiv.org/pdf/1508.04025.pdf)

# Contact 
For any questions or inquiries, feel free to reach out to us:

- Email: [ait-abdelouahab.mehdi@ine.inpt.ac.ma](mailto:ait-abdelouahab.mehdi@ine.inpt.ac.ma)

- LinkedIn: [MEHDI AIT-ABDELOUAHAB](https://www.linkedin.com/in/mehdi-ait-abdelouahab-588481317/)

Thank you for visiting our project repository.
