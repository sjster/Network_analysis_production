
Running the code
-----------------

The project uses the library 'bashful' to execute the spark code from a 'workflow.yaml' YAML file. This file has the file run command and a tag to identify this command.

Link to bashful - https://github.com/wagoodman/bashful

Run the following within the code folder, since folder paths are relative to this folder.

1. To run all tasks - bashful run workflow.yaml
2. To run a specific task named 'build' - bashful run workflow.yaml --tags build

The folder structure
--------------------

1. data
       /temp-data
       /production-data
2. code
      /temp-code
      /production-code
3. output (FOR FILES)
4. output-graphs (FOR FIGURES) 


Execute the code in /code

Cloud storage
-------------

Push the input and output data to Wasabi. The code and configuration information will be in Wasabi_data 


