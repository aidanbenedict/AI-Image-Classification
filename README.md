# AI Image Classification Project

This project implements an AI-based image classification system using Python. The system is capable of performing various image classification tasks using deep learning models.

## Project Structure

- `ai.py`: Main implementation file containing the AI classification logic
- `test.py`: Test file for running and validating the model
- `GARBAGE-CLASSIFICATION-3-2/`: Directory containing classification-related resources (not included in repo)
- `yolov8n.pt`: YOLOv8 model weights (not included in repo)

## Requirements

- Python 3.x
- YOLOv8
- Other dependencies as specified in the code

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/aidanbenedict/AI-Image-Classification.git
   cd AI-Image-Classification
   ```

2. Install the required dependencies:
   ```bash
   pip install ultralytics
   ```
   This will automatically handle the YOLOv8 model weights when needed.

3. Download the training dataset:
   - Request access to the GARBAGE-CLASSIFICATION-3-2 dataset from your project teammates
   - Place the dataset folder in the project root directory

4. Test images:
   - Place your test images in the project root directory
   - Supported formats: .jpg, .jpeg, .png, .webp

## Usage

```python
python test.py  # To run the test script
python ai.py    # To run the main AI classification
```

## Missing Files
The following files are not included in the repository due to size limitations:
1. `yolov8n.pt` - Will be automatically downloaded when running the code
2. `GARBAGE-CLASSIFICATION-3-2/` - Contact team members for access to the dataset
3. Test images - Use your own images or request the test set from team members

## License

This project is open source and available under the MIT License. 