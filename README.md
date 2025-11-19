# Extracting-Monosemanticity-in-Latent-Space-Recommender-Systems

This repository contains code for extracting monosemantic neurons- interpretable latent dimensions- from recommender system embeddings 
using a Sparse Autoencoder (SAE) framework. Our approach leverages a prediction-aware reconstruction loss that propagates gradients
through a frozen recommender to preserve user-item interaction semantics, enabling actionable interventions such as content filtering 
and targeted promotion.

The repository hosts all the code corresponding to our experiments with recommender systems- specifically, Matrix Factorization (MF) and
Neural Collaborative Filtering (NCF)- coupled with Sparse Autoencoder interpretability techniques as described in our article.

## Overview
Modern recommender systems rely on latent embeddings to deliver scalable, personalized predictions. However, such representations are 
inherently opaque and challenging to interpret. In our work, we address these issues by:
- Extracting Monosemantic Neurons: Using Sparse Autoencoders to recover individual latent dimensions that align with meaningful concepts.
- Prediction-Aware Reconstruction: Introducing a novel loss that backpropagates through a fixed recommender model to guarantee that the reconstructed 
  embeddings retain the same recommendation behavior.
- Practical Interventions: Enabling post hoc operations such as content filtering and targeted promotions without retraining the underlying recommender.

The code in this repository implements these ideas for two popular recommendation paradigms:
- Neural Collaborative Filtering (NCF)
- Matrix Factorization (MF)<br>

On two diverse datasets:
- MovieLens M1 (ML1M)
- Last.FM


Below is our architecture diagram, which illustrates how the components of our framework interact:

![Method Diagram](./method_diagram.png)


## Folders
- **code**: Contains all the Jupyter notebooks that implement our method.
  - **data_processing.ipynb**: Prepares and preprocesses raw datasets.
  - **models.ipynb**: Defines the architectures of the recommender systems and the SAE.
  - **training.ipynb**: Contains the training routines for both recommender systems and the SAE.
  - **utils.ipynb**: Provides helper functions for data loading, preprocessing, and evaluation.
  - **visualization_SAE_ncf.ipynb**: Visualizes and evaluates the SAE applied to the NCF model.
  - **visualization_SAE_mf.ipynb**: Visualizes and evaluates the SAE applied to the MF model.
  - **lastFM_notebook.ipynb**: Contains a comprehensive results overview for the Last.FM dataset.

**Important Note**: For ease of use, please move all files from the code folder into the main repository folder (where this README file is located).

- **dataset**: Contains files required to run the experiments, including the ML1M and Last.FM datasets files.
- **res_csv**: Holds CSV output files produced by the recommenders, generated during the training and evaluation of existing models.This folder is 
			   updated whenever a new model is trained and evaluated.
- **models**: Stores the pretrained SAE and recommender system models.
**Important**: Some files in folder **'dataset'** and **'models'**, **'res_csv'** are too large for GitHub and are hosted externally.
				Download the missing dataset files from our [drive folder](https://drive.google.com/drive/folders/12FLL-gItcmZDbEodSjxxd9tfGeQzhXmD?usp=sharing) 
				and place them into the corresponding folder in the repository.

  
## Requirements
The following packages (with the specified versions) were used during development and are required for this project. This project was developed using Python 3.11.11.

The required packages are:
- **torch**: 2.6.0+cu124
- **pandas**: 2.2.2
- **numpy**:2.0.2
- **scipy**:1.14.1
- **seaborn**:0.13.2
- **ipynb**:0.5.1
- **plotly**:5.24.1
- **pathlib**: 1.0.1
- **scikit-learn**:1.6.1
- **matplotlib**:3.10.0

Other modules (e.g., `os`, `itertools`, `math`, `heapq`, `pickle`, etc.) are part of Python's standard library.

To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```


## Usage
1. **Data and Model Files**<br>
	The files in the dataset and models folders are necessary for the proper execution of the code. Before running any code, please download the missing files from the [drive folder](https://drive.google.com/drive/folders/12FLL-gItcmZDbEodSjxxd9tfGeQzhXmD?usp=sharing) and place them in their corresponding folders:
	-Download the dataset subfolder from the drive and merge its contents into your local dataset folder.
	-Download the models subfolder from the drive and merge its contents into your local models folder.
	-Download the models subfolder from the drive and merge its contents into your local res_csv folder.

2. **Notebooks Overview**
- **Evaluating and Visualizing Results:**
  The primary workflow involves loading pre-trained models and running the 
  evaluation notebooks:
  - visualization_SAE_ncf.ipynb (for NCF-based models)
  - visualization_SAE_mf.ipynb (for MF-based models)
  - lastFM_notebook.ipynb (for model performance on the Last.FM dataset)
  
  These notebooks import the necessary variables, functions, and models 
  from other parts of the code, and they generate both quantitative and 
  qualitative results as detailed in the research article.
- **Training a New Model:**
  The training.ipynb notebook is designed for users who wish to train a new model from scratch. This notebook uses the architectures defined in       
  models.ipynb and the preprocessing routines from data_processing.ipynb.
  The default workflow centers on utilizing the pre-trained models for 
  evaluation.
3. **File Organization for Convenience**:<br>
  For optimal ease of maintenance and execution, when working with a particular dataset, please move all files from its corresponding 
  subfolder under code/ into the main directory (where the README file is located). This ensures that the notebooks can locate the 
  necessary scripts without modifying file paths.