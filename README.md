# Nemasis
Machine learning tool for detecting and counting nematode eggs from microscopy images, built as a simple standalone python script.

This tool was developed in tandem by Breeding Insight, a USDA-funded initiative based at Cornell University, and the USDA-ARS sweetpotato breeding program. Details on the nematode experiments from which images were drawn, the imaging process, annotation of the eggs, model training, and model validation will be described in a forthcoming publication.

Please direct any questions to Tyr Wiesner-Hanks ([tw372@cornell.edu](mailto:tw372@cornell.edu))

To get started, download this repository under the `< > CODE` dropdown at top right or clone it using `git clone`.

## SETUP
This code is designed to run from within a [Conda](https://anaconda.org/anaconda/conda) environment. We strongly recommend using [mamba](https://mamba.readthedocs.io/), a much faster implementation of the command-line `conda` tool. All `mamba` commands have the same structure and arguments as the relevant `conda` commands.

To install and set up `mamba`:
1. Download the appropriate installer and complete the installation process for your system
2. Open the terminal and confirm that mamba is installed with `mamba -h`
3. Create a new mamba environment from the YAML file:
   
`mamba env create -f mamba_env_nemasis.yml`

4. Try activating the environment:

`mamba activate nemasis`

6. Run the script using the instructions below. When finished, you can simply close out of the terminal or deactivate the environment:

`mamba deactivate`


## RUNNING THE PIPELINE
Each time you run the pipeline, you will need to activate the `mamba` environment from your command-line environment:

`mamba activate nemasis`


Once the environment is activated, you are ready to run the script. To list the potential commands:

`python nemasis.py -h`


Nemasis can analyze a single image, a directory of images, or a directory output by microscopy software with subdirectories of format `XY01`, `XY02`, etc. Images must of format .jpg, .jpeg, .tif, .tiff, or .png (case insensitive).


`python ./nemasis.py -i sample_images/pmer_37.tif`

`python ./nemasis.py -i sample_images/`


The simplest way to check how Nemasis is performing on your images is to save and inspect annotated output images. It is good practice to save this output and inspect it regularly. You can save this output by specifying a destination with the `-a` flag:

`python ./nemasis.py -i sample_images/ -a sample_annotations/`

The weights for the deep learning model used by Nemasis are stored in `weights.pt`, which the tool will use by default. You can specify your own weights with the `-w` flag:

`python nemasis.py -i sample_images/ -w new_weights.pt`

This makes it fairly simple to update your model with new annotation data or to compare multiple models on a single dataset.

*Note: The new model weights must specify a valid YOLO model, as this tool relies on the `ultralytics` library for the core steps. The target object class name must be `egg`, though this could be easily modified if you're comfortable with python*

## QUESTIONS/COMMENTS  
Please address all questions to Tyr Wiesner-Hanks ([tw372@cornell.edu](mailto:tw372@cornell.edu))

