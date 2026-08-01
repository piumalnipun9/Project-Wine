import json
import sys

def py_to_ipynb(py_file, ipynb_file):
    with open(py_file, 'r') as f:
        code = f.read()

    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.split("\n")]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(ipynb_file, 'w') as f:
        json.dump(notebook, f, indent=2)

if __name__ == '__main__':
    py_to_ipynb('wine_quality_analysis.py', 'wine_quality_analysis.ipynb')
