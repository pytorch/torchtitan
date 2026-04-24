import numpy as np
import cv2

class FisheyeCamera:
    def __init__(
        self,
        # Fisheye image shape
        width=None, height=None,
        # Focal length and pixel skewness
        fx=None, fy=None, skew=None,
        # Principal points (sometimes px, py or cx, cy)
        #px=914.0, py=620.0, #px is cropped sometimes for some reason
        # flipped:
        px=None, py=None, # flipped parameters 910,328
        # Radial distorton coefficients (sometimes k1-k4)
        r1=1.40, r2=0.12, r3=2.20, r4=0.78,
        # Tangential distortion coefficients (sometimes p1-p2 or t1-t2)
        p1=0.0, p2=0.0,
        # Inverse radial distortion coefficients
        inv_r1=-1.1576639, inv_r2=-0.0083861, inv_r3=-1.9612741, inv_r4=0.7758459,
        # Inverse tangential distortion coefficients
        inv_p1=-0.0000021, inv_p2=-0.0000008,
    ):  
        # Fill default values
        if width is None:
            self.width=1792
            sx = 1
        else:
            self.width = width
            sx = width/1792
        if height is None:
            self.height=948
            sy = 1
        else:
            self.height=height
            sy = height/948
        self.fx = np.asarray(1256.0*sx if fx is None else fx)
        self.fy = np.asarray(1256.0*sy if fy is None else fy)
        self.skew = np.asarray(0.0*sx if skew is None else skew)
        self.px = np.asarray((1824-914)*sx if px is None else px) # flipped parameters 910,328
        self.py = np.asarray((948-620)*sy if py is None else py)
        
        self.r1, self.r2, self.r3, self.r4 = r1, r2, r3, r4
        self.p1, self.p2 = p1, p2
        self.inv_r1, self.inv_r2, self.inv_r3, self.inv_r4 = inv_r1, inv_r2, inv_r3, inv_r4
        self.inv_p1, self.inv_p2 = inv_p1, inv_p2

        self._px_to_center = (self.width/2  - self.px)/self.fx
        self._py_to_center = (self.height/2 - self.py)/self.fy
        #print(f"Fisheye width/2 {self.width//2}, px {self.px}, px_to_center {self._px_to_center}, height/2 {self.height//2}, py {self.py}, py_to_center {self._py_to_center}")

        self.K = np.array([[self.fx, self.skew, self.px],
                           [    0.0,   self.fy, self.py], 
                           [    0.0,       0.0,     1.0] ])
        self.dcoeffs = np.array([self.r1, self.r2, self.p1, self.p2, 0.0, self.r3, self.r4, 0.0])
        self._eye3 = np.eye(3, dtype=self.K.dtype)
    
    def __repr__(self):
        return (f"FisheyeCamera(image_width={self.width}, "
               +f"image_height={self.height}, fx={self.fx}, fy={self.fy}, "
               +f"skew={self.skew}, px={self.px}, py={self.py}, "
               +f"r1={self.r1}, r2={self.r2}, r3={self.r3}, r4={self.r4}, "
               +f"p1={self.p1}, p2={self.p2})")

    def real_world_to_fisheye(self, u, v):
        """ Converts from world coordinates to image coordinates """
        u = u #+ self._px_to_center
        v = v #+ self._py_to_center
        radius = (u ** 2 + v ** 2)
        radial_dist = (self.r2 * (radius ** 2) + self.r1 * radius + 1) / (
                self.r4 * (radius ** 2) + self.r3 * radius + 1)
        
        dist_u = u * radial_dist + self.p2 * (3 * u ** 2 + v ** 2) + self.p1 * 2 * u * v
        dist_v = v * radial_dist + self.p1 * (u ** 2 + 3 * v ** 2) + self.p2 * 2 * u * v
        
        x = dist_v * self.skew + self.fx * dist_u + self.px
        y = self.fy * dist_v + self.py
        
        return x, y
    
    def fisheye_to_real_world(self, x, y):
        """ Converts from image coordinates to world coordinates """
        x,y = np.array(x), np.array(y)
        u = x / self.fx - self.skew / (self.fx * self.fy) * y + (self.py * self.skew / self.fy - self.px) / self.fx
        v = y / self.fy - self.py / self.fy

        radius = (u ** 2 + v ** 2)

        numerator = self.inv_r2 * (radius ** 2) + self.inv_r1 * radius + 1
        denominator = self.inv_r4 * (radius ** 2) + self.inv_r3 * radius + 1
        radial_dist = np.true_divide(numerator, denominator, out=np.full(numerator.shape, np.inf), where=denominator != 0)

        dist_u = u * radial_dist + self.inv_p2 * (3 * u ** 2 + v ** 2) + self.inv_p1 * 2 * u * v
        dist_v = v * radial_dist + self.inv_p1 * (u ** 2 + 3 * v ** 2) + self.inv_p2 * 2 * u * v

        return (dist_u, #-self._px_to_center, 
                dist_v, #-self._py_to_center,
                1)
    
    def project_points(
        self,
        image,
        points,
        colors=None,
        sizes=None,
    ):
        u = np.true_divide(points[:, 0], points[:, 2], out=np.full(points[:, 0].shape, np.inf), where=points[:, 2] != 0)
        v = np.true_divide(points[:, 1], points[:, 2], out=np.full(points[:, 1].shape, np.inf), where=points[:, 2] != 0)
        x,y = self.real_world_to_fisheye(u, v)
        
        num_points = len(points)
        # Broadcast color
        if colors is None:
            colors = [[255, 0, 0]] * num_points
        elif isinstance(colors[0], (int, float)):
            colors = [colors] * num_points
        elif len(colors) != num_points:
            raise ValueError(f"Length of 'colors' must match number of points "+
                             f"(got points {points.shape}, "+
                             f"colors {colors.shape if hasattr(colors, 'shape') else colors}, "+
                             f"sizes {sizes.shape if hasattr(sizes, 'shape') else sizes})") # type: ignore[attr-defined]
        colors = np.asarray(colors)
        if colors.max() < 1.05:
            colors = (colors*255).astype(np.uint8)
        # Broadcast size
        if sizes is None:
            sizes = [1] * num_points
        elif isinstance(sizes, (int, float)):
            sizes = [sizes] * num_points
        elif len(sizes) != num_points:
            raise ValueError("Length of 'sizes' must match number of points")
        
        for xi, yi, color, size in zip(x, y, colors, sizes):
            if 0 <= xi < image.shape[1] and 0 <= yi < image.shape[0]:
                color = [int(c) for c in color]
                image = cv2.circle(image, (int(xi), int(yi)), max(1,int(size)), color, -1)
        return image
        
    def project_lines(
        self,
        image,
        lines_3d, # list of lines, each line is a list of 3D points
        color=[255,255,255],
        thickness: int=1,
        alpha: float=1.0,
    ):
        full_line_img = np.copy(image)
        for line in lines_3d:
            pts_2d = []
            for point in line:
                if point[2] > 0:
                    u = point[0] / point[2]
                    v = point[1] / point[2]
                    x,y = self.real_world_to_fisheye(u, v)
                    pts_2d.append( (int(x), int(y)) )
            for i in range(len(pts_2d)-1):
                cv2.line(full_line_img, pts_2d[i], pts_2d[i+1], color, thickness)
        return cv2.addWeighted(image, 1.0-alpha, full_line_img, alpha, 0)

    #@staticmethod
    #def to_hom_coords(pts):
    #    """converts the coordinates to homogeneous (appends ones)"""
    #    return np.concatenate([pts, np.ones(pts.shape[:-1] + (1,), dtype=pts.dtype)], axis=-1)

    #def fisheye_to_real_world_prec(self, x, y):
    #    """ Converts from image coordinates to world coordinates """
    #    img_pts = np.array([[x, y]])
    #    pts_shape = img_pts.shape
    #    img_pts_ = np.ascontiguousarray(img_pts.reshape(-1, 2))
    #    criteria = (cv2.TermCriteria_MAX_ITER | cv2.TermCriteria_EPS, 20, 1e-5)
    #    ud_img_pts_ = cv2.undistortPointsIter(img_pts_, self.K, self.dcoeffs, self._eye3, self._eye3, criteria)
    #    ud_img_pts = ud_img_pts_.reshape(*pts_shape)
    #    dir_vecs = self.to_hom_coords(ud_img_pts)
    #    return dir_vecs[0]


class PinholeCamera:

    def __init__(self, width, height, params={}, 
                 hFov=90., vFov=75., 
                 rel_x_centre=1/2, rel_y_centre=1/2,
                 fisheye_cam=None):
        # if fisheye cam is given, it will match the central points of both models
        
        # Pinhole image shape
        self.width = int(width)    # 3584
        self.height = int(height)  # 1896

        # Focal length
        self.fx = float( params.get('fx', 1256.0) if hFov is None else 
                         self.width/2/np.tan(np.radians(hFov/2)) )
        self.fy = float( params.get('fy', 1256.0) if vFov is None else 
                         self.height/2/np.tan(np.radians(vFov/2)) )
        
        self.skew = float(params.get('skew', 0.0))

        # Principal point x, y axis (image centre)
        if fisheye_cam is None:
            self.px = float( params.get('cx', 2 * 896.0) if rel_x_centre is None else 
                             self.width * rel_x_centre )
            self.py = float( params.get('cy', 2 * 620.0) if rel_y_centre is None else 
                        self.height * rel_y_centre )
        else:
            u, v, _ = fisheye_cam.fisheye_to_real_world(
                fisheye_cam.width/2, 
                fisheye_cam.height/2,
            )
            # that would be mapped to 
            # (fx*u + px + skew*v, fy*v + py)
            # for this to be (width*rel_x_centre, height*rel_y_centre) we solve
            # fx*u + px + skew * v = width/2   =>  px = width/2 - fx*u - skew *v
            # fy*v + py            = height/2  =>  py = height/2 - fy*v
            self.px = float(self.width*rel_x_centre  - self.fx*u - self.skew *v)
            self.py = float(self.height*rel_y_centre - self.fy*v)
            
        #print(f"Pinhole width/2 {self.width//2}, px {self.px}, height/2 {self.height//2}, py {self.py}")
        
        self.K = np.array([[self.fx, self.skew, self.px],
                           [    0.0,   self.fy, self.py], 
                           [    0.0,       0.0,     1.0] ])
        self.dcoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def __repr__(self):
        return (f"PinholeCamera(width={self.width}, height={self.height}, "
                f"fx={self.fx}, fy={self.fy}, skew={self.skew}, "
                f"px={self.px}, py={self.py})")

    def pinhole_to_real_world(self, x, y):
        """ Converts from image coordinates (x, y) to world coordinates (u, v, 1) """
        u = (x - self.px) / self.fx
        v = (y - self.py) / self.fy

        if self.skew:
            u -=  self.skew * v / self.fx
        
        return u, v, 1

    def real_world_to_pinhole(self, u, v):
        """ Converts from world coordinates (u, v, 1) to image coordinates (x, y) """
        #print(f"self.fx {self.fx} ({type(self.fx)}), self.px {self.px} ({type(self.px)}), u {u} ({type(u)})")
        x = self.fx * u + self.px
        if self.skew:
            x += self.skew * v
        y = self.fy * v + self.py
        
        return x, y
        
    def project_points(
        self,
        image,
        points,
        colors=None,
        sizes=None,
    ):
        u = np.true_divide(points[:, 0], points[:, 2], out=np.full(points[:, 0].shape, np.inf), where=points[:, 2] != 0)
        v = np.true_divide(points[:, 1], points[:, 2], out=np.full(points[:, 1].shape, np.inf), where=points[:, 2] != 0)
        x,y = self.real_world_to_pinhole(u, v)
        
        num_points = len(points)
        # Broadcast color
        if colors is None:
            colors = [[255, 0, 0]] * num_points
        elif isinstance(colors[0], (int, float)):
            colors = [colors] * num_points
        elif len(colors) != num_points:
            raise ValueError(f"Length of 'colors' must match number of points "+
                             f"(got points {points.shape}, "+
                             f"colors {colors.shape if hasattr(colors, 'shape') else colors}, "+
                             f"sizes {sizes.shape if hasattr(sizes, 'shape') else sizes})") # type: ignore[attr-defined]
        # Broadcast size
        if sizes is None:
            sizes = [1] * num_points
        elif isinstance(sizes, (int, float)):
            sizes = [sizes] * num_points
        elif len(sizes) != num_points:
            raise ValueError("Length of 'sizes' must match number of points")
        
        for xi, yi, color, size in zip(x, y, colors, sizes):
            if 0 <= xi < image.shape[1] and 0 <= yi < image.shape[0]:
                color = [int(c) for c in color]
                image = cv2.circle(image, (int(xi), int(yi)), int(size), color, -1)
        return image

    def project_lines(
        self,
        image,
        lines_3d, # list of lines, each line is a list of 3D points
        color=[0,0,0],
        thickness:int=1,
        alpha:float=1.0,
    ):
        full_line_img = np.copy(image)
        for line in lines_3d:
            pts_2d = []
            for point in line:
                if point[2] > 0:
                    u = point[0] / point[2]
                    v = point[1] / point[2]
                    x,y = self.real_world_to_pinhole(u, v)
                    pts_2d.append( (int(x), int(y)) )
            for i in range(len(pts_2d)-1):
                cv2.line(full_line_img, pts_2d[i], pts_2d[i+1], color, thickness)
        return cv2.addWeighted(image, 1.0-alpha, full_line_img, alpha, 0)

class Undistorter:
    
    def __init__(
        self,
        pinhole_cam_params,
        fisheye_cam_params,
        pinhole_rel_to_fisheye=True,
        grid_size=20,
        compute_valids=False,
        compute_grids=False,
    ):
        self.grid_size = grid_size
        self.compute_valids = compute_valids
        self.compute_grids = compute_grids
        self.pinhole_rel_to_fisheye = pinhole_rel_to_fisheye
        self.set_cameras(pinhole_cam_params=pinhole_cam_params,
                         fisheye_cam_params=fisheye_cam_params,
                         pinhole_rel_to_fisheye=self.pinhole_rel_to_fisheye)
    
    def set_cameras(
        self,
        pinhole_cam_params=None,
        fisheye_cam_params=None,
        pinhole_rel_to_fisheye=None,
    ):
        pinhole_rel_to_fisheye = pinhole_rel_to_fisheye or self.pinhole_rel_to_fisheye
        cameras_changed = False

        if fisheye_cam_params is not None:
            self.fisheye_cam = FisheyeCamera(**fisheye_cam_params)
            cameras_changed = True
        if pinhole_cam_params is not None:
            if pinhole_rel_to_fisheye:
                self.pinhole_cam = PinholeCamera(**pinhole_cam_params, 
                                                 fisheye_cam=self.fisheye_cam)
            else:
                self.pinhole_cam = PinholeCamera(**pinhole_cam_params)
            cameras_changed = True
        if cameras_changed:
            cols = np.arange(start=0, stop=self.pinhole_cam.width, step=1, dtype=np.float32)
            rows = np.arange(start=0, stop=self.pinhole_cam.height, step=1, dtype=np.float32)
            x, y = np.meshgrid(cols, rows)
            (u, v, w) = self.pinhole_cam.pinhole_to_real_world(x, y)
            self.p2f_map_x, self.p2f_map_y = self.fisheye_cam.real_world_to_fisheye(u, v)

            cols = np.arange(start=0, stop=self.fisheye_cam.width, step=1, dtype=np.float32)
            rows = np.arange(start=0, stop=self.fisheye_cam.height, step=1, dtype=np.float32)
            x, y = np.meshgrid(cols, rows)
            (u, v, w) = self.fisheye_cam.fisheye_to_real_world(x, y)
            self.f2p_map_x, self.f2p_map_y = self.pinhole_cam.real_world_to_pinhole(u, v)
            self.f2p_map_x = self.f2p_map_x.astype(np.float32)
            self.f2p_map_y = self.f2p_map_y.astype(np.float32)

            if self.compute_valids:
                self.undistorted_pinhole_valid = self.undistort_fisheye_to_pinhole(
                    np.ones((self.fisheye_cam.height, self.fisheye_cam.width), dtype=np.uint8)
                ) > 0
                self.undistorted_fisheye_valid = self.undistort_pinhole_to_fisheye(
                    np.ones((self.pinhole_cam.height, self.pinhole_cam.width), dtype=np.uint8)
                ) > 0
            if self.compute_grids:
                grid = np.zeros((self.pinhole_cam.height, self.pinhole_cam.width, 3), dtype=np.uint8)
                grid[::self.grid_size, :, 0] = 255
                grid[:, ::self.grid_size, 0] = 255
                self.fisheye_grid = self.undistort_pinhole_to_fisheye(grid)

                grid = np.zeros((self.fisheye_cam.height, self.fisheye_cam.width, 3), dtype=np.uint8)
                grid[::self.grid_size, :, 2] = 255
                grid[:, ::self.grid_size, 2] = 255
                self.pinhole_grid = self.undistort_fisheye_to_pinhole(grid)

    def undistort_fisheye_to_pinhole(self, fisheye_img):
        return cv2.remap(fisheye_img, self.p2f_map_x, self.p2f_map_y, cv2.INTER_LINEAR)

    def undistort_pinhole_to_fisheye(self, pinhole_img):
        return cv2.remap(pinhole_img, self.f2p_map_x, self.f2p_map_y, cv2.INTER_LINEAR)
    
    def project_points(
        self,
        image,
        points,
        colors=None,
        sizes=None,
        pinhole=False,
    ):
        num_points = len(points)

        # Broadcast color
        if colors is None:
            colors = [[255, 0, 0]] * num_points
        elif isinstance(colors[0], (int, float)):
            colors = [colors] * num_points
        elif len(colors) != num_points:
            raise ValueError(f"Length of 'colors' must match number of points "+
                             f"(got points {points.shape}, "+
                             f"colors {colors.shape if hasattr(colors, 'shape') else colors}, "+
                             f"sizes {sizes.shape if hasattr(sizes, 'shape') else sizes})") # type: ignore[attr-defined]

        # Broadcast size
        if sizes is None:
            sizes = [1] * num_points
        elif isinstance(sizes, (int, float)):
            sizes = [sizes] * num_points
        elif len(sizes) != num_points:
            raise ValueError("Length of 'sizes' must match number of points")

        # Project 3D points to 2D
        image_points, _ = cv2.projectPoints(
            objectPoints=points,
            rvec=np.zeros(3),
            tvec=np.zeros(3),
            cameraMatrix=self.pinhole_cam.K if pinhole else self.fisheye_cam.K,
            distCoeffs=self.pinhole_cam.dcoeffs if pinhole else self.fisheye_cam.dcoeffs,
        )
        if False:
            p = 7
            print(f"\n fx {self.pinhole_cam.fx if pinhole else self.fisheye_cam.fx}"+
                    f" px {self.pinhole_cam.px if pinhole else self.fisheye_cam.px}"+
                    f" fy {self.pinhole_cam.fy if pinhole else self.fisheye_cam.fy}"+
                    f" py {self.pinhole_cam.py if pinhole else self.fisheye_cam.py}")
            for i in range(points.shape[0]):
                if image_points[i,0,0] < 0 or image.shape[1] <= image_points[i,0,0]:
                    print("out of lateral bounds: ", end="")
                if image_points[i,0,1] < 0 or image.shape[0] <= image_points[i,0,1]:
                    print("out of vertical bounds: ", end="")
                print(f"[{points[i,0]:{p}.2f}, {points[i,1]:{p}.2f}, {points[i,2]:{p}.2f}] -> [{image_points[i,0,0]:{p}.2f}, {image_points[i,0,1]:{p}.2f}]")
            

        # Draw points
        for point, color, size in zip(image_points, colors, sizes):
            x, y = point[0]
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                color = [int(c) for c in color]
                image = cv2.circle(image, (int(x), int(y)), int(size), color, -1)

        if True:
            pass
        elif pinhole:
            #print(f"image shape {image.shape}, py {self.pinhole_cam.py} fy {self.pinhole_cam.fy}, py+fy {self.pinhole_cam.py+self.pinhole_cam.fy}, py-fy {self.pinhole_cam.py-self.pinhole_cam.fy}")
            image = cv2.circle(image, (int(self.pinhole_cam.px), int(self.pinhole_cam.py)), 5, [0, 255, 0], 5)
            image = cv2.circle(image, (int(self.pinhole_cam.px+self.pinhole_cam.fx), int(self.pinhole_cam.py)), 5, [0, 255, 0], 5)
            image = cv2.circle(image, (int(self.pinhole_cam.px-self.pinhole_cam.fx), int(self.pinhole_cam.py)), 5, [0, 255, 0], 5)
            image = cv2.circle(image, (int(self.pinhole_cam.px), int(self.pinhole_cam.py+self.pinhole_cam.fy)), 5, [0, 255, 0], 5)
            image = cv2.circle(image, (int(self.pinhole_cam.px), int(self.pinhole_cam.py-self.pinhole_cam.fy)), 5, [0, 255, 0], 5)
        else:
            #print(f"image shape {image.shape}, py {self.fisheye_cam.py} fy {self.fisheye_cam.fy}, py+fy {self.fisheye_cam.py+self.fisheye_cam.fy}, py-fy {self.fisheye_cam.py-self.fisheye_cam.fy}")
            image = cv2.circle(image, (int(self.fisheye_cam.px), int(self.fisheye_cam.py)), 5, [0, 255, 0], 5)
            image = cv2.circle(image, (int(self.fisheye_cam.px+self.fisheye_cam.fx), int(self.fisheye_cam.py)), 5, [0, 255, 0], 5)
            image = cv2.circle(image, (int(self.fisheye_cam.px-self.fisheye_cam.fx), int(self.fisheye_cam.py)), 5, [0, 255, 0], 5)
            image = cv2.circle(image, (int(self.fisheye_cam.px), int(self.fisheye_cam.py+self.fisheye_cam.fy)), 5, [0, 255, 0], 5)
            image = cv2.circle(image, (int(self.fisheye_cam.px), int(self.fisheye_cam.py-self.fisheye_cam.fy)), 5, [0, 255, 0], 5)

        return image
