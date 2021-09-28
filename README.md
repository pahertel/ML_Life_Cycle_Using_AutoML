# ML_Life_Cycle_Using_AutoML_With_Hyperparameter_Optimization_Focused_On_Costs_With_Data_Drift

### Summary:
Project shows the machine learning life cycle using auto ML to find the best economic model and hyperparameters meeting certain business requirements then deployed using batch inferencing checking for data drift by looking at default of credit card clients from Taiwan in 2005 from UCI Machine Learning Repository: default of credit card clients Data Set. This project will go over the objectives listed below

### Classification AutoML Objectives:
1. Inspect and Visualize data
2. Set experiment to record all metrics in MLflow and a local JSON file
3. Creating an AutoML function with hyperparameter optimization
4. Visualize all model metrics
5. Analyze models based on business costs
6. Save best model separately
7. Visualize the best model
8. Explainable Artificial Intelligence with Random Forest and LIME

### Batch Inference Objectives:
1. Check for data drift using the Two Sample Kolmogorov Smirnov Test
2. Retrieve the best model saved from the default of credit card classification AutoML from Taiwan in 2005
3. Make predictions on new loan data
4. Explainable Artificial Intelligence with Random Forest and LIME
5. Saving the predictions and logging artifacts by creating a new experiment in mlflow

### Production and Scaling:
1.	Dockerfile to create docker image for sharing with team or deploy on, HPC, the cloud or Kubernetes


### Business Scenario:
Train a sample of the training data to estimate which model and hyperparameters to use based on cost for final training meeting these requirements below.

- Minimum precision: .3
- Minimum recall: .3
- Type 1 error cost: $50
- Type 2 error cost: $35
- Run time per hour: $3
- Projected number of loans: 300,000
- Projected file size in GB: 3 GB


### How to use Docker assuming you have Docker Desktop downloaded already:
1. Create Docker image from Dockerfile:
	- Go to the Command Line or Bash
	- Go to directory of Dockerfile
	- Name and build docker image by typing and giving it a name
		- docker build -t <image_name> .

2. Run docker image:
	- In a command Line or Bash run
		- docker run -p 8888:8888 <image_name>
	- Then paste the URL in a browser

### Libraries Used:
- Plotly and seaborn are used for visualization
- Scipy for Two Sample Kolmogorov Smirnov Test
- Pandas
- Numpy
- MLflow to make experiments and record data
- scikit-learn
- Lime to explain individual predictions 


# ROC Curve:
![image](https://user-images.githubusercontent.com/71287557/135013326-692a2972-85af-4e67-858d-ca9e11c6786f.png)

# Confusion Matrix:
![image](https://user-images.githubusercontent.com/71287557/135013346-0cb48ada-0db5-479f-9fcf-f9601e3ad0fa.png)

# Explainable AI using Random Forest for Global Predictions and LIME to Explain Individual Predictions:
![image](https://user-images.githubusercontent.com/71287557/135013370-779bb7cb-0d45-4b32-91f9-f391d2f70357.png)
![image](https://user-images.githubusercontent.com/71287557/135013383-10cfd7f8-be3e-43ae-9d6c-ac2f677962b5.png)

# MLflow Experiments:
![image](https://user-images.githubusercontent.com/71287557/135013404-56196608-76c9-453a-95d7-79745358eed2.png)
![image](https://user-images.githubusercontent.com/71287557/135013416-578e9318-4d46-41a9-8c59-63b575364011.png)
![image](https://user-images.githubusercontent.com/71287557/135013431-242de8ff-0c9a-435c-b4c1-64d43a15680b.png)
![image](https://user-images.githubusercontent.com/71287557/135013441-e196b9fa-0bea-437d-9f15-84702bc83826.png)
