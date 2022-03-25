# Main Workshop Notebook
1. From the [data folder](https://github.com/zbutton314/workshop-classification/tree/main/data), download spambase.csv and spambase_val.csv
  - Click file, click "Raw", right-click and Save As, add the ".csv" extension
3. From the [notebook file](https://github.com/zbutton314/workshop-classification/blob/main/notebooks/Classification_Walkthrough.ipynb), click "Open in Colab"
4. In Google Colab, upload the two CSV files
5. Use Shift+Enter to execute a cell (model building code near bottom may take a few minutes)

# Resources
- [Workshop Slides](https://docs.google.com/presentation/d/1lQHQxkNJnh-mQr_F8ezILrz44P1MWel1e7JkUiFdnQc/edit#slide=id.p)
- Model/Algorithm Explanations
  - [Naive Bayes](https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c)
  - [K-Nearest Neighbors](https://towardsdatascience.com/knn-k-nearest-neighbors-1-a4707b24bd1d)
  - [Logistic Regression](https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148)
  - [Decision Trees](https://towardsdatascience.com/the-complete-guide-to-decision-trees-28a4e3c7be14)
  - [Random Forest](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)
  - [XGBoost](https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7)
  - [Support Vector Machines](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)

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
