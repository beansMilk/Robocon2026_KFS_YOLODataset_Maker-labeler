# Robocon2026_KFS_YOLODataset_Maker-labeler

A Python project based on OpenGL and Pygame to make & label datasets for several kinds of YOLO models (detect, seg, pose)

---

## Project Structure

```
Robocon2026_KFS_YOLODataset_Maker-labeler/
├── texture file/          # Texture images (.png)
│                           # Note: textures may not be identical to those in the game, but they are used
├── origin file/           # Initial script for generating KFS and saving images
├── dataset_maker/         # Scripts for automatic label generation
└── README.md
```

---

## Important Notice

**Labels are automatically generated only when each image is created.**

**If you lose the label file, you must label it manually.**

---

## Features

- Generate images with automatic YOLO-format labels
- Support YOLO detection, segmentation, and pose estimation
- OpenGL rendering with Pygame display
- Customizable textures, lighting, and camera angles

---

## Requirements

- Python 3.8+
- Pygame
- PyOpenGL
- NumPy

Install dependencies:

```bash
pip install pygame PyOpenGL numpy
```

---

## Output Format

Generated dataset structure:

```
output/
├── images/          # Rendered images
│   ├── 0001.png
│   └── ...
└── labels/          # YOLO format labels
    ├── 0001.txt
    └── ...
```

---

## Acknowledgments

- DS & GPT: Assistance in understanding OpenGL and script customization
- YT_X: Help with random lighting implementation and support for more dataset formats

---
# Statement

## Statement

#### The author is currently a student with a busy academic schedule. As such, the maintenance of this project is not yet systematic and may appear somewhat rough. Your understanding and tolerance would be greatly appreciated.

#### Special thanks to NEUQ for its strong support throughout the ROBOCON. We sincerely hope to receive the university's recognition and further support in the future.

## License

MIT License

Copyright (c) 2026 beansMilk@github.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.