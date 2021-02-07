# DSC180 Malware Detection 

## Introduction 
As the internet techniques are growing at a fast speed nowadays, people are starting to worry about their data safety. Since many of us will store our important information on our cellphones, we need to find an appropriate way to secure our cell phone away from malwares. It is also large companies’ like Google who offer a third open market to ensure the data security of their customers. Therefore, more researchers are participating in the research area of detecting malwares. There is a new method called HinDroid, which finds out the relationships between applications for malware detection. However, while using the HinDroid, some shortages of the model have been found. For example, the HinDroid only uses parts of the features contained in the smali file. But there are more useful features that can help detect malwares after doing analysis. There are also other models which relate to NLP that are popular in the malware detection research domain. Therefore, in this report, we will compare the performance of HinDroid, Word2vec, Node2vec, and Metapath2vec.

## Data Generating Process
### Data Source 
The data source is called AnDroid Malware Dataset (AMD). The dataset is published in 2017 by the Argus Lab from the University of South Florida. This data source is used by many other malware detection papers and widely used in the research domain. 
### Data Description 
The original source is the APK(Android Application Package), which can be decompiled by Apktool. After converting the APK package into different files, we select smali files specifically for detecting malwares. Smali files are a type of file converted from the original java code. The researchers considered that the malicious actions are contained in the smali file and it is more meaningful to use smali files. We randomly choose 200 malwares and 200 benigns from the dataset.
### Smali File 
In order to analyze the smali files, we should understand the structure of it. Therefore, here is the description of the smali files and the features contained in the smali files.
![smali](data/report/smali.png)
