# MatLabProj

1. Installation

	In order to run the project, the following Matlab Toolboxes must be installed: 

		1. Statistics and Machine Learning Toolbox

		2. Deep Learning Toolbox

		3. Parallel Computing Toolbox

        4. Text Analytics Toolbox

        5. Text Analytics Toolbox Model for fastText English 16 Billion Token Word Embedding 

        6. Bioinformatics Toolbox

2. Instruction - training:

    Step1: Change the configuration in modeConfig.m
    
    Step2: Run preprocessing.m to load the data and preprocessing the data. 

    Step3: Train the model with train.m

3. Instruction - prediting: If you would like to test the model without training
    
    Step1: Set configuration in modelConfig.m to cfg.execMode = "test", choose the model (lstm, bilstm, cnn) for loading pretrained model

    Step2: Run preprocessing.m

    Step3: Run train.m