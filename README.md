# diffusion_MNIST

Training difusion model to generate MNIST digits.

Two main approaches used here - conditional and unconditional models. For conditional one I also embed the label so that the model knows which digit exactly it is reconstructing.

evaluation.ipynb - jupyter notebook were FID is calculated and examples of digits are generated

model_cond.py, model_uncond.py - models that are used (UNet)

training_cond.ipynb, training_uncond.ipynb - Jupyter notebooks for training the models

LeNet - the network that is used for FID evaluation.
