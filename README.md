# MatLabProj

1. Installation

	In order to run the project, the following Matlab Toolboxes must be installed: 

		1. Statistics and Machine Learning Toolbox

		2. Deep Learning Toolbox

		3. Parallel Computing Toolbox

        4. Text Analytics Toolbox

        5. Text Analytics Toolbox Model for fastText English 16 Billion Token Word Embedding 

        6. Bioinformatics Toolbox

2. Folder structure

	preprocess: Folder contains function to preprocess text before inputting to the model

	evaluation: Folder contains function to evaluate the performance of the model

	model: Folder contains model scripts

	pretrained: Folder contains our pretrained model, it is usually used for predict without training

	save_models: Folder contains your model after training

	datasets: Folders contains different datasets for training and testing. There are two different set of datasets. 

	- AdHocAnnouncements dataset is the original dataset from this paper https://arxiv.org/pdf/1710.03954.pdf

	- daset2Label*: is the dataset we create on the similar method but in different periods, * represents for the way to build the dataset
                    

3. Instruction - training:

    Step1: Change the configuration in modeConfig.m
    
    Step2: Run preprocessing.m to load the data and preprocessing the data. 

    Step3: Train the model with train.m

4. Instruction - prediting: If you would like to test the model without training
    
    Step1: Set configuration in modelConfig.m to cfg.execMode = "test", choose the model (lstm, bilstm, cnn) for loading pretrained model

    Step2: Run preprocessing.m

    Step3: Run train.m
