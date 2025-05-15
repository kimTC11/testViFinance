## Data Preprocess

We use the [Defects4J methods provided by A3Test](https://github.com/awsm-research/A3Test/tree/main/Defects4j%20Method). Use the following scripts for data preprocessing and context information extraction:

```sh
cd data
python data_sampling.py
python global_information.py
python reverse.py
```

Before preprocessing, ensure Defects4J and tree-sitter are correctly installed. You can also download our processed data directly.

You can also directly download our processed data here.
