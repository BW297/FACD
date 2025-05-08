# FACD

The implementation for the IJCAI-25 paper "A Fast-Adaptive Cognitive Diagnosis Framework for Computerized Adaptive Testing Systems". The main paper and the appendix are in the main.pdf and appendix.pdf respectively, which are contained in the folder named paper.

# 💻 Requirements	

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

# 🛠️ Experiments

You can run our framework in provided dataset with BECAT selection strategy using following command:

```shell
bash scripts/run.sh
```

If you want to change the CAT selection strategy, just replace the parameter of `--strategy` with the name of your target strategy, such as random, MAAT, NCAT.

# Reference

Yuanhao Liu, Yiya You, Shuo Liu, Hong Qian, Ying Qian, Aimin Zhou "A Fast-Adaptive Cognitive Diagnosis Framework for Computerized Adaptive Testing Systems" In Proceedings of the 34th International Joint Conference on Artificial Intelligence, 2025.

## Bibtex
```
@inproceedings{liu2025facd,
 author = {Yuanhao Liu and Yiya You and Shuo Liu and Hong Qian and Ying Qian and Aimin Zhou},
 booktitle = {Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI)},
 title = {A Fast-Adaptive Cognitive Diagnosis Framework for Computerized Adaptive Testing Systems},
 year = {2025},
 address = {Montreal, Canada}
}
```
