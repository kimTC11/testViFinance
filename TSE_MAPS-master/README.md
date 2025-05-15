# TSE replication package
This repository contains the replication package for our TSE submission "The Prompt Alchemist: Automated LLM-Tailored Prompt Optimization for Test Case Generation".


## Dependency
Our work requires Java 1.8 and Defects4J. Please follow the setup instructions in the [Defects4J repository](https://github.com/rjust/defects4j/tree/master) to ensure the correct usage of the `defects4j` command.


### Additional dependency 
Run the following command in the root directory of this repository:

```sh
pip install -r requirements.txt
```


## Usage

1. First use the ``defects4j checkout -p`` command to export each project from Defects4J and generate test cases of evosuite with ``gen_tests.pl``.

2. Our code relies on OpenAI (for ChatGPT) and DeepInfra (for Llama-3.1 and Qwen2) services. Obtain their API keys and add them to `code/auto_prompt_bsl.py` and `code/auto_prompt_ours.py` 

3. We use the [Defects4J methods provided by A3Test](https://github.com/awsm-research/A3Test/tree/main/Defects4j%20Method). Use the following scripts for data preprocessing and context information extraction:

```sh
cd data
python data_sampling.py
python global_information.py
python reverse.py
```

Before preprocessing, ensure Defects4J and tree-sitter are correctly installed. You can also download our processed data directly.

You can also directly use our processed data in the ``data`` folder.

4. Run the folllow command and modify the setting in ``run_bsl.sh`` to reproduce the results of each baseline method.

```sh
cd code
bash run_bsl.sh
```

5. Run the folllow command and modify the setting in ``run_bsl.sh`` for our method. 

```sh
cd code
bash run_ours.sh
```

## Benchmarks
We provide the detailed information of the used benchmark in the ``benchmark.md`` file.


## Results
We provide more detailed results in the ``result.md`` file.
