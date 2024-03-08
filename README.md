# Bi-SGTAR
 Official implementation of "Bi-SGTAR: A Simple yet Efficient Model for CircRNA-Disease Association Prediction based on Known Association Pair Only", accepted by Knowledge-Based Systems.

In recent years, research has intensively pursued predicting circRNA-disease associations, often favoring complex models integrating multiple data sources and advanced neural architectures. Conversely, our work explores whether traditional ML techniques can achieve equivalent performance. Comparative experiments with 25 state-of-the-art models affirm Bi-SGTAR's effectiveness. Notably, Bi-SGTAR's minimal data requirement, utilizing only a **adjacency matrix**, is advantageous for researchers with **limited resources**. 

Given the low data requirements of the model, there is promising potential for its application in various fields related to non-coding RNA (ncRNA)-disease or drug association identification. And our initial investigations on lncRNA-disease and microbe-drug datasets have been encouraging. We thus encourage you, the researchers, to test the performance of Bi-SGTAR in other non-coding RNA-disease, drug-target, and other association prediction areas. Since it only requires a known association matrix, this is very simple to implement.

# Training on Colab
Open Pub-Bi-SGTAR.ipynb and click the ["Open in Colab"](https://colab.research.google.com/github/Shiy-Li/Bi-SGTAR/blob/main/Pub_Bi_SGTAR.ipynb) button to quickly reproduce the results in the Google Colab environment.

# Cite
If you compare with, build on, or use aspects of this work, please cite the following:

```js/java/c#/text
@article{li2024Bi-SGTAR,
  title={Bi-SGTAR: A Simple yet Efficient Model for CircRNA-Disease Association Prediction based on Known Association Pair Only},
  author={Li, Shiyuan and Chen, Qingfeng and Liu, Zhixian and Pan, Shirui and Zhang, Shichao},
  journal={Knowledge-Based Systems},
  year={2024},
  publisher={Elsevier}
}
```
