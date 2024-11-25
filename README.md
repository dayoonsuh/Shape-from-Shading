# Photometric Stereo
Shape from shading involves reconstructing the 3D shape of an object from a single 2D image using variations in shading. The task leverages the relationship between surface orientation, light source direction, and image intensity to estimate the surface normals and, ultimately, the object's geometry.
![image](https://github.com/user-attachments/assets/cd8e88d1-a7c6-446e-a7e6-1a2e25179884)
![image](https://github.com/user-attachments/assets/cf18945a-da35-469d-a689-53032130ec84)

![image](https://github.com/user-attachments/assets/75f6573b-00c7-41a3-bcdc-f4e547ffd30a)


## Requirements
```python
matplotlib==3.8.4
numpy==2.1.3
opencv_python==4.10.0.84
Pillow==11.0.0
scipy==1.14.1
skimage==0.0
torch==2.4.0
torchvision==0.19.0
```

### Run
```python
python main.py -s [subject name] -i [integration method]
```
- subject name: {yaleB01, yaleB02, yaleB05, yaleB07}
- integration method: {row, column, average, random}

### Change Patch size, Number of iterations, Threshold
change image_sticher.py 
