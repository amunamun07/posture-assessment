# Thesis Project: Workout Assessment Using Yolov7-Pose

Agriculture makes the people less dependent on foreign countries as it provides food and also provides income to the farmers and revenue to the government. 
This project is initiated to help these farmers with suggestions about the best crop that can be grown in their land.

The project objectives are:
- From the given dataset, we need to analyze the different conditions where crops are grown.
- We need to group the list of crops based on the similar condition that they grow in.
- We also need to predict which crop can be grown with which given conditions.
- Creating a monitoring system that can monitor the often updating data to create a better model.


💫 **Version 1.0**
## 📖 Documentation

| Documentation              |                                                                |
| -------------------------- | -------------------------------------------------------------- |
| 📚 **[Project Details]**   | Document on Objective, System Workflow, Outputs and limitations| 
| 📚 **[Project Slides]**    | Final Workflow and Demo Presentation                           |

[Project Details]: 
[Project Slides]: 

## Features

- Training yolov7-Pose 
- 


## 📦 Setting up the project
```bash
git clone git@github.com:amunamun07/posture-assessment.git
```

You need to set up the environment.

- **Operating system**: Linux (Ubuntu 20.04)
- **Python version**: Python 3.9 (only 64 bit)
- **Package managers**: [docker] & [docker-compose]

[docker]: https://www.docker.com/
[docker-compose]: https://docs.docker.com/compose/

### pipenv

Using pip, we can install the pipenv package manager, make sure that
your `pip`, `setuptools` and `wheel` are up-to-date. If you already have 
pipenv installed, you can skip the first command.

```bash
pip install pipenv
```
To create a pipenv environment in a folder.
```bash
pipenv shell
```
To get the project requirements.
```bash
pipenv sync
```

### Getting the dataset

To download the dataset into the folder, use:
```bash
wget https://drive.google.com/file/d/1WCpTGIJrHudvWEbRe09dDIAYKJo2GiHo/view?usp=sharing crop_data.csv
```

### Getting the model

To download the model for ease, use:
```bash
wget https://drive.google.com/file/d/1q5wvhOCLYTSJxp6R8iRgsAfBnsNay44n/view?usp=sharing model.plk
```
(Alternatively): Copy and open these links and download them manually.

## ⚒ Run the project

We need to activate both the servers streamlit and flask at once. So one of the way is to run streamlit
in the background and then running the flask app.
Alternatively you can use two subprocesses and manage running both the servers.

```bash
# Run the Streamlit Dashboard in the background
streamlit run dashboard/main.py &

# (Optional) -> run the Streamlit Dashboard only
streamlit run dashboard/main.py

# Run the Flask Server
python app.py
```

## 🚦 Run tests

You can run the test files using the command below

```bash
pytest -v
