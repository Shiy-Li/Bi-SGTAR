# Bi-SGTAR
 Official implementation of "Bi-SGTAR: A Simple yet Efficient Model for CircRNA-Disease Association Prediction based on Known Association Pair Only", accepted by Knowledge-Based Systems.

In recent years, research has intensively pursued predicting circRNA-disease associations, often favoring complex models integrating multiple data sources and advanced neural architectures. Conversely, our work explores whether traditional ML techniques can match predictive performance. Notably, Bi-SGTAR's minimal data requirement, utilizing only a basic data-adjacency matrix, is advantageous for researchers with limited resources. Comparative experiments with 21 state-of-the-art models affirm Bi-SGTAR's effectiveness. 
Given the low data requirements of the model, there is promising potential for its application in various fields related to non-coding RNA (ncRNA)-disease or drug association identification. While our initial investigations on lncRNA-disease and microbe-drug datasets have been encouraging, it is crucial to recognize that the current success of Bi-SGTAR in these specific domains does not necessarily imply its seamless efficiency in diverse association prediction tasks across multiple domains. 
We encourage you, the researchers, to test the performance of Bi-SGTAR in other non-coding RNA-disease, drug target, and other association prediction areas. Since it only requires a known association matrix, this is very simple to implement. 

# Training on Colab
Open Pub-Bi-SGTAR.ipynb and click the "Open in Colab" button to quickly reproduce the results in the Google Colab environment.

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
