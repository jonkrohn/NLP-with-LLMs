# Sample Instructions for LLM Fine-Tuning on Multiple LLMs

1. Use [Paperspace](https://www.paperspace.com/) for cloud training:
     * Select **Core** virtual servers
     * **Create A Machine** with the following configuration:
          * **OS**: ML-in-a-Box 
          * **Machine Type**: select cheapest Multi-GPU option
          * **Region**: closest to you
          * **Authentication**: select `Password`
          * In **Advanced Options**, give your machine a name and select `Static` Public IP
     * Wait for machine to boot
 2. In a command-line interface, use `ssh` address and corresponding password provided by Paperspace
     * Run `git clone https://github.com/jonkrohn/NLP-with-LLMs.git`
     * If using [Poetry](https://python-poetry.org/) for the first time, run `curl -sSL https://install.python-poetry.org | python3 -`
     * Change into the repo's directory with `cd NLP-with-LLMs/`
     * If the repo's dependencies haven't already been installed with Poetry, run `poetry install`
     * Fine-tune the T5 LLM with `poetry run python code/Finetune-T5-multiGPU.py`
3. In a separate command-line window (that's also SSH'ed into your Paperspace instance), you can confirm multiple-GPU usage with `nvidia-smi`
4. When you are satisfied with your model, you can push the model to Hugging Face:
     * Uncomment these lines in `Finetune-T5-multiGPU.py`:
          * `training_model.model.push_to_hub("digit_conversion")`
          * `training_model.tokenizer.push_to_hub("digit_conversion")`
     * Run `poetry run huggingface-cli login`
