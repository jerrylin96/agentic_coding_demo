# Agentic Coding for Science Demo

This GitHub repository contains code for a demo on agentic coding for science. The associated slides can be found here:
[Agentic Coding for Science Slides](https://docs.google.com/presentation/d/1AvQg3AVCsG52QoDZwgQ3cTqrW1SsQVa8fjQ_EcEu5-M/edit?usp=sharing)

## Contents

- `demo_notebook.ipynb`: A Google Colab notebook demonstrating how to train a convolutional neural network (CNN) on the MNIST dataset using JAX and the Flax NNX API. It is optimized for Cloud TPUs. This is based off the [official Flax MNIST tutorial](https://flax.readthedocs.io/en/latest/mnist_tutorial.html).
- `bonsai/`: A git submodule linking to the [JAX Bonsai repository](https://github.com/jax-ml/bonsai).

## Instructions

1. **Fork the repository**: Click the "Fork" button in the top right and navigate to your new fork.

2. **Clone your fork and setup submodules**: Clone your forked repository to your local machine:
   ```bash
   git clone --recursive <your-fork-url> <desired-path-here>
   ```

3. **Open Google Antigravity**: If you do not have the agy shortcut installed, simply choose open folder within Google Antigravity and navigate to the path you cloned to.
   ```bash
   agy <desired-path-here>
   ```

4. **Run `demo_notebook.ipynb` on TPU**:
   - Open `demo_notebook.ipynb` within Antigravity.
   - Ensure the Colab session is connected to a Google Colab TPU runtime.
   - Run the setup cells to install the necessary packages (`jax[tpu]`, `flax`, `tensorflow`, `jaxlib`) and restart the kernel when prompted.
   - The notebook will automatically mount your Google Drive to save data and output models.

5. **Understand codebase and iterate with Gemini Code Assist and Agent Manager**:
   - Use Gemini Code Assist on the left panel to tag the notebook and ask questions about it.
   - Inside the notebook, use `cmd + I` (for mac) or `ctrl + I` (for windows) in individual cells to add additional documentation where unclear.
   - Use the Agent Manager (top right) to ask the agent to come up with a plan to create a new, separate version of `demo_notebook.ipynb` that uses an advanced architecture from bonsai, keeping in mind that the Google Colab TPU kernel can only see inside the Google Drive folder and that bonsai was git cloned in that folder earlier.
   - Comment on the plan where necessary and allow the agent to execute once you are aligned.
   - Continue to iterate as needed.

