
## To run

### Python

create the Python environment using anaconda (or miniconda, micromamba, etc):

    conda env create -f environment.yml
    conda activate testenv

or if on Linux or Apple, you can optionally:  
- install direnv: https://direnv.net/docs/installation.html  
- hook it into your shell: https://direnv.net/docs/hook.html  
- then `cd` into this directory and type `direnv allow` to create and install a local microconda environment  

There's also a vscode plugin called `direnv` which will actiavte the environment in vscode.




### Dotnet

run:

    dotnet publish -c Release cuda-hack

to download CUDA and make it available in .net interactive / polyglot notebooks
