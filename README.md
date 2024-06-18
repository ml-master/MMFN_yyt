# MMFN-fake-news-detection"# MMFN" 
the implement for "Multi-modal Fake News Detection on Social Media via Multi-grained Information Fusion" by PyTorch.

Enviroment requirements are as follows:
| enviroment      | version |
| ----------- | ----------- |
| pytorch      | 1.13.1       |
| cuda   | 12.2        |
| python   | 3.9.19        |
# About
The dataset includes three datasets: **weibo, twitter, gossipcop, and gossipcop-LLM**. According to the original author's intention, only the dataset link of **gossipcop-LLM** is given here:
https://github.com/junyachen/Data-examples

Other datasets can be obtained by contacting the original author.

The py file with the suffix of preprocess indicates the preprocessing of the dataset, and the py file with the suffix of dataset indicates the dataset class.

**trainMMFN.py** is the training code. Running this code can get the reproducible results. Modifying the **forward** function in **class MMFN** can get different ablation experiment results.
# Result
The overall expriments as follows:
![Overall](整体比较.png)

My reproduce expriments as follows:
![Overall](复现结果.png)

The abalation expriments as follows:
![Overall](消融实验.png)

