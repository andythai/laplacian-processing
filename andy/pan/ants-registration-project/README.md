# ANTs Registration Project

This project demonstrates how to use ANTs (Advanced Normalization Tools) to register a moving image to a fixed image. It also includes functionality to transform an array of corresponding points to ensure they map correctly to the registered image.

## Project Structure

```
ants-registration-project
├── src
│   ├── register.py          # Main logic for image registration
│   ├── transform_points.py   # Function to transform corresponding points
│   └── utils.py             # Utility functions for loading and saving data
├── data
│   ├── moving_image.nii.gz   # The moving image to be registered
│   ├── fixed_image.nii.gz    # The fixed image for registration
│   └── points.npy            # Array of corresponding points in the moving image
├── requirements.txt          # List of dependencies for the project
└── README.md                 # Documentation for the project
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd ants-registration-project
pip install -r requirements.txt
```

## Usage

1. Place your moving and fixed images in the `data` directory.
2. Update the paths in `src/register.py` if necessary.
3. Run the registration script:

```bash
python src/register.py
```

4. The registered image will be saved in the specified output path within the script.
5. To transform the corresponding points, run:

```bash
python src/transform_points.py
```

## Dependencies

- ANTsPy
- NumPy
- nibabel
- Other necessary libraries listed in `requirements.txt`

## License

This project is licensed under the MIT License. See the LICENSE file for more details.