# scripts README
---

This folder contains scripts that process raw data to make intermediate data and or figures.

They should contain none to minimal private functions that load or process data. Most code should be imported from the modules created above. The primary function of these scripts is to apply the functinos in the above modules to the data in an organized and static way.

When run, these scripts should generate a folder in the results folder that has a descritpive name, and the date the script was run (e.g., manifolds_2023-10-31). The date should be in YYYY-MM-DD format.

Scripts should always be run in the directory they live in to allow for the relative import of the reset of the package.

Scripts should start with these lines:
```
import sys # This allows us to add the parent directory in the next line
sys.path.append('..') # Add the parent directory which allows for imports within the package
```
The code will likely continue with these lines, or something like them to import the project packages.
`from src import io,proc`

Use of the `click` package is preferred when making scripts which is very useful for including command line arguments.
This prevents having to modify python files to run the same code on multiple datasets
https://click.palletsprojects.com/en/8.1.x/

There is an example of this kind of script called `example_script_click.py`