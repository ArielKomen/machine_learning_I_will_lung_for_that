<h2>Image analysis project</h2>

Authors:

- Cas van Rijbroek
- Lex Bosch
- Ariël Komen
- Anton Ligterink

DISCLAIMER: most of the code in this repo has been extracted from https://github.com/gregwchase/nih-chest-xray. We modified the main CNN model file to accept command line arguments to make it suitable for grid computing.

X-ray images were cropped and converted to numpy arrays using the files in this directory. In addition to this, the labels of the images were reduced from 709 to 15 categories. Irrelivant columns and patients with ages not represented in years were also removed (this methodoligy is identical to the one used by the researchers who publicised the original repository).

All python code used in our analysis is present in this repository.
