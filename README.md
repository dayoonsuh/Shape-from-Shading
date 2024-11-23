# Shape-from-Shading

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
