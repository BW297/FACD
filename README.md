# FACD</u>-IJCAI 2025

We provide comprehensive instructions on how to run FACD in the ***<u>"scripts"</u>*** directory. 

# Requirements	

```python
dgl==2.1.0+cu121
numpy==2.1.2
pandas==2.2.3
scikit_learn==1.4.1.post1
scipy==1.14.1
torch==2.2.1+cu121
torch_geometric==2.6.1
torch_sparse==0.6.18+pt22cu121
tqdm==4.65.0
vegas==6.0.1
```
Please install all the dependencies listed in the `requirements.txt` file by running the following command:

```bash
pip install -r requirements.txt
```

# Experiments

Firstly, you need

> cd scripts

Then, create a ***<u>"ckpt"</u>*** directory in ***<u>"mdoel"</u>*** directory and you can run our framework in different dataset with random selection strategy using following command:

```shell
bash run.sh
```

If you want to change the CAT selection strategy, just replace the parameter of `--strategy` with the name of your target strategy, such as MAAT, NCAT, BECAT.
