# CKM
CKM: Compact Knowledge Memory for Continual Anomaly Detection
# Abstract
Continual anomaly detection (CAD) extends traditional anomaly detection to dynamic production environments, requiring models to identify anomalies in new product categories without forgetting previously learned normal patterns.
Existing CAD methods typically share the same parameter across all categories, which may cause cross-class interference and weaken the ability of the model to capture fine-grained features essential for detecting subtle anomalies.
To address this issue, we propose a Compact Knowledge Memory (CKM) for CAD. 
Unlike conventional approaches, CKM disentangles task-specific knowledge by projecting the learned parameters into a compact latent space, where the class-specific coefficient matrices are independently preserved.
This compact representation not only alleviates parameter redundancy but also minimizes interference across categories. 
Moreover, to enhance the modeling of fine-grained normal patterns, CKM integrates a refiner module and employs a dynamic weighted cosine loss, further improving feature discrimination and ensuring stable performance in continual anomaly detection. 
Experiments show that CKM significantly outperforms existing methods, achieving state-of-the-art performance in CAD.
# Overview of CKM
![Uploading CKM.png…]()
# Train an
d Test
We test datasets on the MVTec AD and VisA.  
The test results will be given after the training is completed
```python 
   python main.py
```
