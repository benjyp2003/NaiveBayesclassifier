# Naive Bayes Classifier

This project implements a Naive Bayes classifier. 
From building the model to classifying data according to the model.


## project versions

this project has 3 versions represented by different branches:

- branch `fastapi` - a fastapi version of the project
- branch `container-v1` - a single container version of the project
- branch `container-v2` - a two-container version of the project

## Project Structure (v1 - v2)

```
app/
  api/          # Contains API endpoints such as /train /classify /test /load_data and more
  controller/   # Handles the control flow between modules
  core/         # Contains the core logic for the Naive Bayes algorithm
  data/         # Data files storage
  ui/           # User interface components (CLI, GUI, or web)
  
.gitignore # Specifies files and directories to ignore in the repository
requirements.txt # Lists the Python dependencies for the project
Dockerfile # Dockerfile for building the application container (v2 only)
```

## project structure (v3)

```
builder_app/       # Contains the builder application/container for training the Naive Bayes classifier
  api/          # Contains API endpoints such as /health /train /test /load_data and more
  controller/   # Handles the control flow between modules
  core/         # Contains the core logic for the Naive Bayes algorithm trining
  data/         # Data files storage
  ui/           # User interface components (CLI, GUI, or web)
 
  .gitignore        # Specifies files and directories to ignore in the repository
  Dockerfile        # Dockerfile for building the application container 
  requirements.txt  # Lists the Python dependencies for the project
  
classifier_app/     # Contains the classifier application/container for classifying data 
    api/          # Contains API endpoints such as /classify /predict and more
    core/         # Contains the core logic for the Naive Bayes algorithm classification
    
    .gitignore        # Specifies files and directories to ignore in the repository
    Dockerfile        # Dockerfile for building the application container 
    requirements.txt  # Lists the Python dependencies for the project
    
```

## Running the Project:

```
v1 - run in cmd 'fastapi dev app/api/server:app'
     and then run the main.py file.
     now the fastapi server is running on http://localhost:8000
```     
```
v2 - run 'docker build -t <image-name> .'
     and then run 'docker run -p 8000:8000 <image-name>'
     now you have a container running the fastapi server on http://localhost:8000
```
```
v3 - first create a common network for the two containers to communicate: 'docker create network <network-name>'
     then run 'docker build -t <builder-image-name> .'
     run 'docker build -t <builder-image-name> .'    
     then 'docker run -d --name building_app_container --network <network-name> -p 8000:8000 <builder-image-name>' 
     then 'docker run -d --name <classifeir-container-name> --network <network-name> -p 8001:8000 <classifier-image-name>' .
     now you have two containers running two fastapi servers on http://localhost:8000 and http://localhost:8001
```

