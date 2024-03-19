# DeePNAPWebsite

# DeePNAP

DeePNAP is a software that provides a web portal for predicting the protein-nucleic acid (PNA) binding affinity solely from their sequences. This task is crucial for experimental design and analysis of PNA interactions (PNAIs). The software is based on a machine learning model trained on a large and diverse dataset of 14,401 entries from the ProNAB database, including wild-type and mutant PNA complex binding parameters.

DeePNAP aims to accurately predict the binding affinity and free energy changes resulting from mutations in PNAIs using only the sequences as input. Unlike other similar tools that utilize both sequence and structure information, DeePNAP relies solely on sequence-based features. This approach yields high correlation coefficients between the predicted and experimental values, with low root mean squared errors for PNA complexes when predicting the log10(KD) values. These results demonstrate the generalizability of DeePNAP.

To validate the model's performance, its predictions were compared with experimentally measured binding affinities of BarA-/BfmR-DNA. The comparison showed excellent agreement, further confirming the accuracy of DeePNAP's predictions.

In addition to the model itself, a web interface has been developed to host DeePNAP. This web portal is a powerful tool for rapidly predicting binding affinities for a wide range of PNAIs with high precision. It provides researchers with a convenient means of gaining insights into the implications of PNAIs in various biological systems.

## Prerequisites

Before running a local instance of the web portal, ensure that you have the following prerequisites installed:

1. **Python**: Make sure you have the latest version of Python installed on your system. You can download it from the official Python website: [python.org](https://www.python.org/).
2. **Required Libraries**: The necessary libraries for running DeePNAP are listed in the `requirements.txt` file. These dependencies need to be installed before running the software.

## Installation Instructions

To install and run DeePNAP on your computer, follow the instructions below:

1. **Clone the repository**: Start by cloning this GitHub repository to a directory on your local machine. You can do this by executing the following command in your terminal or command prompt:
    
    ```bash
    git clone https://github.com/StructuralBiologyLabIISERTirupati/DeePNAPWebsite.git
    ```
    
2. **Create a Python virtual environment**: It is recommended to create a Python virtual environment to isolate the dependencies for DeePNAP. Open a terminal or command prompt and navigate to the cloned repository's directory. Then, create a virtual environment by running the following commands:
    
    ```bash
    cd DeePNAPWebsite
    python -m venv env
    ```
    
    This will create a new directory called `env` that contains the isolated Python environment.
    
3. **Activate the virtual environment**: Activate the virtual environment you just created. The activation steps depend on your operating system:
    - **Windows**:
        
        ```bash
        env\Scripts\activate
        ```
        
    - **MacOS/Linux**:
        
        ```bash
        source env/bin/activate
        ```
        
4. **Install dependencies**: With the virtual environment activated, install the required libraries listed in the `requirements.txt` file. Run the following command:
    
    ```bash
    pip install -r requirements.txt
    ```

5. **Import the Checkpoint Files**: Once the dependencies are installed, import the checkpoint files into the correct location. The files can be downloaded from the following link: 
https://drive.google.com/drive/folders/1BPPCl_izq96_MD60FXx60slgY8oPb-Qa?usp=sharing

    Make sure the checkpoint files are in the folder checkpoint_files
    
6. **Run the software**: Once the dependencies are installed and the checkpoint files are imported, you can run DeePNAP by executing the `__init__.py` file. Run the following command:
    
    ```bash
    python __init__.py
    ```
    

This will start the local instance of the web portal, and you should be able to access it in your web browser at the appropriate address provided in the console.

Congratulations! You have successfully installed and launched DeePNAP on your computer. You can now utilize the web portal on your local Machine.
