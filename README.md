This is the project pipeline for building a Multimodal Image Captioning and Assistive Q&A System, designed to generate textual descriptions from images and answer related questions. The system is particularly geared toward supporting visually impaired users by interpreting visual content through AI.

Project Type : Deep Learning, Computer Vision, Natural Language Processing  
Core Tasks : Image Captioning, Visual Question Answering (VQA), Multimodal Understanding  
Dataset Used : MS-COCO 2017 

1) Project Overview

The primary goal of this project is to build an intelligent multimodal system capable of understanding and describing visual content through natural language. Specifically, it aims to generate accurate and context-aware captions for images and enable users to interact with the system by asking questions about those images.This functionality has direct applications in accessibility tools for visually impaired individuals, automated content generation, smart surveillance, and human-computer interaction. The project combines computer vision and natural language processing techniques to bridge the gap between visual understanding and language generation.

2) Dataset Description:  MS-COCO 2017
Images: 118,000+ training images (`train2017`)
Annotations : Over 414,000 captions
Format : Each image is associated with 5 human-written captions.

3)Prerequisites
Python 3.7 or above
pip or conda package manager
Jupyter Notebook
MS-COCO 2017 Dataset (captions + images)

4)Data Preprocessing
The data preprocessing phase prepares the MS-COCO 2017 dataset for training a deep learning-based image captioning system. This involves parsing annotations, validating image availability, cleaning and formatting captions, and generating tokenized sequences compatible with deep learning models.

1. Loading and Parsing Annotations
The preprocessing begins by loading the official captions_train2017.json file from the MS-COCO dataset. This file contains image IDs and associated human-written captions. Each image in the training set typically has five captions. The code constructs a dictionary mapping each image ID to a list of its corresponding captions.

2. Caption Cleaning and Formatting
To standardize the input text for modeling, each caption is cleaned and normalized. This includes converting all text to lowercase, removing excess whitespace, and appending special tokens: startseq at the beginning and endseq at the end of every caption. These tokens help the sequence model recognize the boundaries of each sentence during training.

3. Validating Local Image Files
The script verifies that each image referenced in the annotation file actually exists in the local directory (train2017/). If an image file is missing, its associated captions are discarded. This step ensures alignment between the text and the visual data, avoiding runtime errors and misaligned training samples.

4. Caption Tokenization
Using Keras’ Tokenizer, the cleaned captions are converted into sequences of integers. Each unique word is mapped to an index, and the tokenizer handles unknown or rare words using a special <unk> token. This results in a vocabulary dictionary (word → index) and a numerical sequence for every caption, enabling compatibility with embedding layers and neural networks.

5. Vocabulary and Sequence Statistics
To support downstream model configuration, the script computes important statistics from the tokenized data. These include the total vocabulary size (i.e., number of unique words), the maximum caption length in tokens, and the number of valid image-caption pairs. For example, after filtering, the dataset typically contains around 82,783 image-caption pairs, a vocabulary of roughly 11,488 tokens, and captions up to 35 tokens in length.



