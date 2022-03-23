# Main Workshop Notebook
1. From the [data folder](https://github.com/zbutton314/workshop-classification/tree/main/data), download spambase.csv and spambase_val.csv
2. From the [notebook file](https://github.com/zbutton314/workshop-classification/blob/main/notebooks/Classification_Walkthrough.ipynb), click "Open in Colab"
3. In Google Colab, upload the two CSV files

# Spam Hunter Package

### Follow these steps to set up your environment
1. Install Anaconda/Miniconda
2. Clone this repository into the directory of your choosing
3. Navigate to your workshop-classification directory in cmd/terminal
  - Ex. "cd path/to/repo/workshop-classification"
4. Create conda virtual environment
  - Ex. "conda env create --name spam --file spam-env.txt"
5. Activate virtual environment
  - Ex. "conda activate spam"

### Follow these steps to run the package
1. Navigate to workshop-classification directory
2. Activate conda virtual environment
3. Execute main.py
  - "python -m spam_hunter.main"
