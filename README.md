# **CV-APP-Assignment #1**
Interactive Computer Vision demo application built with OpenCV.

Includes multiple modes: 
1. Convert image color between RGB ↔ Grey ↔ HSV  
2. Contrast and brightness adjustment  
3. Show image histogram  
4. Gaussian filter with changeable parameter  
5. Bilateral filter with changeable parameters  
6. Canny edge detection  
7. Line detection using Hough Transform  
8. Create a panorama 
9. Image translation, rotation, and scale  
10. Camera calibration  
11. Augmented Reality with TREX model projection

----------------------------------------------------
## **Setup & Run the Application**
1. Clone the repository.
   ```bash 
   https://github.com/APS2542/CV-Assignment.git
2. Into project directory.
   ```bash
   cd CV-Assignment
4. Create a virtual environment.
   ```bash
   uv venv .venv
5. Activate the environment
   ```bash
   source .venv/bin/activate   # Linux / macOS
   .venv\Scripts\activate      # Windows PowerShell
7. Install dependencies.
   ```bash
   uv pip install -r requirements.txt
8. Run the application
   ```bash
   python app.py 
-------------------------------------------------------------------
## **Mode Controls**
- Press key 0 to Normal Mode
- Press keys 1–9 to switch between other modes
- Press + to run Camera Calibration
- Press - to run Augmented Reality (AR)
- Press ESC to quit
------------------------------------------------------------------
## **Mode Overview**
| Key   | Mode Description                                   |
| ----- | -------------------------------------------------- |
| **0** | Normal Mode (raw webcam feed)                      |
| **1** | Convert image color (RGB ↔ Gray ↔ HSV)             |
| **2** | Contrast & Brightness adjustment                   |
| **3** | Image Histogram                                    |
| **4** | Gaussian Filter (parameter changeable)             |
| **5** | Bilateral Filter (parameter changeable)            |
| **6** | Canny Edge Detection                               |
| **7** | Hough Transform for Line Detection                 |
| **8** | Panorama Stitching                                 |
| **9** | Image Translation, Rotation, Scaling               |
| **+** | Camera Calibration (saves `calibration.npz`)       |
| **-** | Augmented Reality (T-rex)                          |

------------------------------------------------------------------
## Aphisit st126130
