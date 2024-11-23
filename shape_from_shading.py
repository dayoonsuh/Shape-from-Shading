"""Implements Shape from shading."""
import time
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision

from helpers import LoadFaceImages, display_output, plot_surface_normals


class ShapeFromShading(object):
    def __init__(self, full_path, subject_name, integration_method):
        ambient_image, imarray, light_dirs = LoadFaceImages(full_path,subject_name,64)
        self.n = 25
        # Preprocess the data
        processed_imarray = self._preprocess(
            ambient_image, imarray) 

        # Compute albedo and surface normals
        albedo_image, surface_normals = self._photometric_stereo(processed_imarray, light_dirs)
        # Save the output
        self.save_outputs(subject_name, albedo_image, surface_normals)
        plot_surface_normals(surface_normals)
        plt.savefig('%s_surface_normal_map.jpg' % (subject_name))

        # Compute height map

        # row
        start_time = time.time()
        height_map = self._get_surface(surface_normals, "row")
        end_time = time.time()
        print(f"Computation time for row: {end_time-start_time} seconds")
        display_output(albedo_image, height_map)
        plt.savefig('%s_height_map_view1_%s.jpg' % (subject_name, "row"))
        # display_output(albedo_image, height_map, view_angle=[-5, -55])
        # plt.savefig('%s_height_map_view2_%s.jpg' % (subject_name, "row"))
        # # display_output(albedo_image, height_map, view_angle=[0, 0])
        # # plt.savefig('%s_height_map_view3_%s.jpg' % (subject_name, "row"))
        # # display_output(albedo_image, height_map, view_angle=[-5, 30])
        # # plt.savefig('%s_height_map_view4_%s.jpg' % (subject_name, "row"))
        plt.close()
        # column
        start_time = time.time()
        height_map = self._get_surface(surface_normals, "column")
        end_time = time.time()
        print(f"Computation time for column: {end_time-start_time} seconds")
        display_output(albedo_image, height_map)
        plt.savefig('%s_height_map_view1_%s.jpg' % (subject_name, "column"))
        # display_output(albedo_image, height_map, view_angle=[-5, -55])
        # plt.savefig('%s_height_map_view2_%s.jpg' % (subject_name, "column"))
        # display_output(albedo_image, height_map, view_angle=[0, 0])
        # plt.savefig('%s_height_map_view3_%s.jpg' % (subject_name, "column"))
        # display_output(albedo_image, height_map, view_angle=[-5, 30])
        # plt.savefig('%s_height_map_view4_%s.jpg' % (subject_name, "column"))
        plt.close()
        # average
        start_time = time.time()
        height_map = self._get_surface(surface_normals, "average")
        end_time = time.time()
        print(f"Computation time for average: {end_time-start_time} seconds")
        display_output(albedo_image, height_map)
        plt.savefig('%s_height_map_view1_%s.jpg' % (subject_name, "average"))
        # display_output(albedo_image, height_map, view_angle=[-5, -55])
        # plt.savefig('%s_height_map_view2_%s.jpg' % (subject_name, "average"))
        # display_output(albedo_image, height_map, view_angle=[0, 0])
        # plt.savefig('%s_height_map_view3_%s.jpg' % (subject_name, "average"))
        # display_output(albedo_image, height_map, view_angle=[-5, 30])
        # plt.savefig('%s_height_map_view4_%s.jpg' % (subject_name, "average"))
        plt.close()
        # quit()

        start_time = time.time()
        height_map = self._get_surface(surface_normals, integration_method)  # argument: random
        end_time = time.time()
        print(f"Computation time for {integration_method}: {end_time-start_time} seconds")

        # Save output results
        display_output(albedo_image, height_map)
        plt.savefig('%s_height_map_view1_%s_%s.jpg' % (subject_name, integration_method, self.n))
        display_output(albedo_image, height_map, view_angle=[-5, -55])
        plt.savefig('%s_height_map_view2_%s_%s.jpg' %(subject_name, integration_method, self.n))
        display_output(albedo_image, height_map, view_angle=[0, 0])
        plt.savefig('%s_height_map_view3_%s_%s.jpg' % (subject_name, integration_method, self.n))
        display_output(albedo_image, height_map, view_angle=[-5, 30])
        plt.savefig('%s_height_map_view4_%s_%s.jpg' % (subject_name, integration_method, self.n))

    def save_input_images(self, subject_name, processed_imarray):
        transform = torchvision.transforms.ToPILImage()
        for i in range(1, processed_imarray.shape[0]+1):
            img_tensor = processed_imarray[i-1]

            # Convert to PIL Image
            img_pil = transform(img_tensor)

            # Save as a JPG file
            img_pil.save(f"{subject_name}_{i}.jpg")

    def _preprocess(self, ambimage, imarray):
        """
        preprocess the data:
            1. subtract ambient_image from each image in imarray.
            2. make sure no pixel is less than zero.
            3. rescale values in imarray to be between 0 and 1.
        Inputs:
            ambimage: h x w
            imarray: Nimages x h x w
        Outputs:
            processed_imarray: Nimages x h x w
        """
        processed_imarray = (imarray - ambimage)/255.
        negative_idx = torch.where(processed_imarray < 0)
        processed_imarray[negative_idx] = 0

        return processed_imarray

    def _photometric_stereo(self, imarray, light_dirs):
        """
        Inputs:
            imarray:  N x h x w
            light_dirs: N x 3
        Outputs:
            albedo_image: h x w
            surface_norms: h x w x 3
        """

        n, h, w = imarray.shape  
        imarray = imarray.reshape(n, h*w)
        g = torch.linalg.lstsq(light_dirs, imarray).solution
        albedo_image = torch.norm(g, dim=0)
        albedo_image = albedo_image.reshape((1, h*w))
        surface_normals = g / albedo_image
        albedo_image = albedo_image.reshape(h, w)
        surface_normals = torch.permute(
            surface_normals.reshape(3, h, w), (1, 2, 0))
        return albedo_image, surface_normals

    def _get_surface(self, surface_normals, integration_method):
        """
        Inputs:
            surface_normals:h x w x 3
            integration_method: string in ['average', 'column', 'row', 'random']
        Outputs:
            height_map: h x w
        """
        print(integration_method)
        h, w = surface_normals.shape[:2]
        fy = surface_normals[:, :, 0] / surface_normals[:, :, 2]
        fx = surface_normals[:, :, 1] / surface_normals[:, :, 2]

        Fy = torch.cumsum(fy, dim=1)  # add towards down # [h, w]
        Fx = torch.cumsum(fx, dim=0)  # add towards right # [h, w]
        if integration_method == "row":  # right then down
            
            FFy = Fy[0].repeat(h, 1)
            height_map = FFy + Fx

        elif integration_method == "column":  # down then right
            FFx = Fx[:, 0].reshape(h, 1).repeat(1, w)
            height_map = FFx + Fy

        elif integration_method == "average":
            FFy = Fy[0].repeat(h, 1)
            row_sum = FFy + Fx
            FFx = Fx[:, 0].reshape(h, 1).repeat(1, w)
            col_sum = FFx + Fy
            height_map = (row_sum+col_sum)/2

        elif integration_method == "random":
            n = self.n
            print(f"Number of path: {n}")
            fx = fx.numpy()
            fy = fy.numpy()
            height_map = np.zeros((h, w))
            for p in range(1, n+1):
                directions = np.array(
                    [np.random.randint(0, 2, 168) for _ in range(192)])
                for x in range(h):
                    for y in range(w):
                        cury, curx, cursum = 0, 0, 0
                        while cury < y or curx < x:
                            if cury == y:  # reached the last row
                                curx += 1
                                cursum += fx[curx, cury]
                            elif curx == x:  # reached last col
                                cury += 1
                                cursum += fy[curx, cury]
                            else:  # choose which dir to go
                                direction = directions[curx, cury]
                                if direction:
                                    curx += 1
                                    cursum += fx[curx, cury]
                                else:
                                    cury += 1
                                    cursum += fy[curx, cury]
                        height_map[x, y] += cursum
            height_map /= n
            height_map = torch.from_numpy(height_map)

        return height_map

    def save_outputs(self, subject_name, albedo_image, surface_normals):
        im = Image.fromarray((albedo_image*255).numpy().astype(np.uint8))
        im.save("%s_albedo.jpg" % subject_name)
        im = Image.fromarray(
            (surface_normals[:, :, 0]*128+128).numpy().astype(np.uint8))
        im.save("%s_normals_x.jpg" % subject_name)
        im = Image.fromarray(
            (surface_normals[:, :, 1]*128+128).numpy().astype(np.uint8))
        im.save("%s_normals_y.jpg" % subject_name)
        im = Image.fromarray(
            (surface_normals[:, :, 2]*128+128).numpy().astype(np.uint8))
        im.save("%s_normals_z.jpg" % subject_name)