# OpenFunsearch

OpenFunsearch is an open-source project aiming to contribute to the scientific community by simplifying the use of the Funsearch algorithm. This project provides a detailed setup guide and easy-to-use scripts for running and analyzing Funsearch experiments.

## Overview

The Funsearch algorithm, originally developed by Google DeepMind, is a cutting-edge method for solving complex search problems. For more information on the original research, please refer to the [Nature paper](https://www.nature.com/articles/s41586-023-06924-6).

You can also find the original Funsearch repository on [GitHub](https://github.com/google-deepmind/funsearch).

We are not the first to try and work on Funsearch, changes have been made my github user: jonppe at [Github](https://github.com/jonppe/funsearch) Our build mostly continues on his work.

## Features

- **Simplified Setup**: Follow the detailed setup guide to install and configure the necessary environment.
- **Command-line Execution**: Run Funsearch experiments easily from the command line using `run_funsearch.py`.
- **Comprehensive Analysis**: Analyze the results of your experiments using the `run_analyzer.ipynb` Jupyter notebook.

## Getting Started

To get started with OpenFunsearch, please refer to the [Setup Guide](setup_guide.md) for detailed installation and setup instructions.

### Running Funsearch

To run a Funsearch experiment, you can use the `run_funsearch.py` script. This script can be executed from the command line and can be altered to take all needed things such as LLM, Problem, Variables etc.

```bash
python run_funsearch.py
```

### Analyzing Results 
After completing a run, you can analyze the results using the run_analyzer.ipynb Jupyter notebook. This notebook provides an exploratory data analysis to get a quick overview over the results. Please refer to the specific eval_datetime.txt for analysis on a specific run.

## Contact
If you have any questions or need further assistance, please feel free to contact me!

#### Disclaimer
This project is written by MsC Computer Science students Remco Stuij and Peter van Driel at Leiden University. Any suggestions or improvements are welcome. 
