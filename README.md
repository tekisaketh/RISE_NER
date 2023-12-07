# RISE - NER Model Training and Evaluation on Babelscape/MultiNERD Dataset


## TASK
The task is to train a model (A) on MultiNERD dataset with all the tags available and another model (B) with few of the tags and then compare them using metrics.

More details about the models can be found on Hugging Face website:

Model A - https://huggingface.co/Saketh/entity-recognition-general-sota-v1-finetuned-ner

Model B - https://huggingface.co/Saketh/entity-recognition-general-sota-v1-finetuned-ner-X

Analysis:

Training & validation set results:
Metric	Model A	Model B
Training Loss	0.0323	0.0214
Validation Loss	0.0396	0.0228
Precision	0.9138	0.9472
Recall	0.9146	0.9621
F1	0.9142	0.9546
Accuracy	0.9857	0.9915

Test set results:
Metric	Model A	Model B
Test Loss	0.027	0.017
Test Precision	0.93	0.95
Test Recall	0.95	0.97
Test F1	0.94	0.96
Test Accuracy	0.990	0.993

As we can see, Model B has better performance than Model A on all metrics. This suggests that Model B is a better model for NER tasks.
The reason for this is likely that Model B is trained on a smaller dataset, which allows it to focus on a more specific set of tags. 
This can lead to better performance on the test set, as Model B is less likely to overfit to the training data.

This is due to 2 reasons:
Smaller dataset: Model B is trained on a smaller dataset of only 11 tags, compared to Model A which is trained on all 31 tags. This means that Model B has fewer parameters to learn, which can help to prevent overfitting.
More focused dataset: The 11 tags that Model B is trained on are likely to be more generic than the 31 tags that Model A is trained on. 
This can make it easier for Model B to learn the relationships between the tags and text data, which can lead to better performance.
Overall, based on the results - Model B is a better model for NER tasks than Model A. 
This is likely due to the fact that Model B is trained on a smaller, more focused dataset, which helps to prevent overfitting and improve performance.

Requirements: python pip modules are present in the notebook itself to install.
I've used google collab and levaraged A100 & V100 GPUs based on availability to train the RoBERTa based pre-trained models.
