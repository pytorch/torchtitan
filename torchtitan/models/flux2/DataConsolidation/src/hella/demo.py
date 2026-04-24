# Taken from https://github.com/nxtAIM/hella_dataloader/blob/main/hella_dataset.py
# on 11.11.2025 21:43

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025-06-30
# @Author  : Zhaoze Wang (FORVIA HELLA)

from hella_dataset import HellaDataset
import os
import json
import numpy as np
import open3d as o3d
import cv2
import time
import argparse
from typing import Dict, List, Tuple, Optional, Any
from numpy import ndarray


class HellaDatasetVisualizer:
    def __init__(self, dataset: HellaDataset, config: Dict[str, Any]) -> None:
        """Initialize the Visualizer with dataset and configuration
        
        Args:
            dataset (HellaDataset): The dataset instance containing point clouds and images
            config (Dict[str, Any]): Configuration dictionary with visualization parameters
        """
        self.dataset: HellaDataset = dataset
        self.config: Dict[str, Any] = config
        self.vis: Optional[o3d.visualization.Visualizer] = None
        self.pcd_geom: Optional[o3d.geometry.PointCloud] = None
        self.car_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.coord: Optional[o3d.geometry.TriangleMesh] = None
        self.writer: Optional[cv2.VideoWriter] = None
        
    def load_pointcloud_from_array(self, points_array: ndarray) -> o3d.geometry.PointCloud:
        """Load point cloud from numpy array
        
        Args:
            points_array (ndarray): Numpy array of shape (N, 3+) containing point coordinates
            
        Returns:
            o3d.geometry.PointCloud: Open3D point cloud object
            
        Raises:
            RuntimeError: If the array is empty or invalid
        """
        try:
            points = points_array[:, :3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
        except Exception as e:
            raise RuntimeError(f"Empty or invalid pointcloud array: {str(e)}")
        return pcd

    def load_car_model(self, path: str) -> o3d.geometry.TriangleMesh:
        """Load and configure car mesh model
        
        Args:
            path (str): File path to the car 3D model (.obj format)
            
        Returns:
            o3d.geometry.TriangleMesh: Configured car mesh with transformations applied
        """
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()
        rotation_x = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
        mesh.rotate(rotation_x, center=mesh.get_center())
        mesh.scale(1.3, center=mesh.get_center())
        mesh.translate([0, 0, -1.3])
        return mesh

    def load_xyz_frame(self, size: float = 1.5) -> o3d.geometry.TriangleMesh:
        """Load coordinate frame
        
        Args:
            size (float): Size of the coordinate frame axes
            
        Returns:
            o3d.geometry.TriangleMesh: Coordinate frame mesh with rotation applied
        """
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        rotation_z = coord.get_rotation_matrix_from_xyz((0, 0, np.pi / 2))
        coord.rotate(rotation_z, center=coord.get_center())
        return coord

    def save_camera_parameters(self, filename: str = "assets/camera.json") -> None:
        """Save current camera view parameters to file
        
        Args:
            filename (str): Output JSON file path for camera parameters
        """
        if self.vis is None:
            return
        ctr = self.vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        data = {
            "intrinsic": {
                "width": params.intrinsic.width,
                "height": params.intrinsic.height,
                "intrinsic_matrix": np.asarray(params.intrinsic.intrinsic_matrix).tolist()
            },
            "extrinsic": np.asarray(params.extrinsic).tolist()
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Camera saved] {filename}")

    def load_camera_parameters(self, filename: str = "assets/camera.json") -> Optional[o3d.camera.PinholeCameraParameters]:
        """Read camera parameters from file
        
        Args:
            filename (str): Input JSON file path containing camera parameters
            
        Returns:
            Optional[o3d.camera.PinholeCameraParameters]: Camera parameters if file exists, None otherwise
        """
        if not os.path.exists(filename):
            return None
        with open(filename, "r") as f:
            data = json.load(f)
        params = o3d.camera.PinholeCameraParameters()
        intr = o3d.camera.PinholeCameraIntrinsic()
        intr.set_intrinsics(
            width=int(data["intrinsic"]["width"]),
            height=int(data["intrinsic"]["height"]),
            fx=float(data["intrinsic"]["intrinsic_matrix"][0][0]),
            fy=float(data["intrinsic"]["intrinsic_matrix"][1][1]),
            cx=float(data["intrinsic"]["intrinsic_matrix"][0][2]),
            cy=float(data["intrinsic"]["intrinsic_matrix"][1][2])
        )
        params.intrinsic = intr
        params.extrinsic = np.array(data["extrinsic"])
        return params

    def set_view_control_topdown(self, center: Tuple[float, float, float] = (0, 0, 0)) -> None:
        """Set default top-down view
        
        Args:
            center (Tuple[float, float, float]): The (x, y, z) coordinates to look at
        """
        if self.vis is None:
            return
        ctr = self.vis.get_view_control()
        ctr.set_lookat(center)
        ctr.set_front([0.0, -1.0, -0.4])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.45)

    def add_text_to_image(self, img: ndarray, text: str, position: Tuple[int, int] = (10, 25)) -> ndarray:
        """Add text with black background to image
        
        Args:
            img (ndarray): Input image array (BGR format)
            text (str): Text string to display
            position (Tuple[int, int]): (x, y) position for text placement
            
        Returns:
            ndarray: Image with text overlay
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)  # Black background
        
        # Get text size
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_w, text_h = text_size
        
        # Create background rectangle
        x, y = position
        cv2.rectangle(img, (x-5, y-text_h-5), (x+text_w+5, y+5), bg_color, -1)
        
        # Add text
        cv2.putText(img, text, position, font, font_scale, text_color, font_thickness)
        
        return img

    def combine_images(self, pointcloud_img: ndarray, camera_imgs: List[Optional[ndarray]]) -> ndarray:
        """Combine point cloud image with 6 camera views (3 on top, 3 on bottom)
        Uses config sizes for all dimensions
        
        Args:
            pointcloud_img (ndarray): Point cloud visualization image
            camera_imgs (List[Optional[ndarray]]): List of 6 camera images (can contain None)
            
        Returns:
            ndarray: Combined image with layout: [3 cams top | PCD center | 3 cams bottom]
        """
        h_pcd, w_pcd, _ = pointcloud_img.shape  # Should match pcd_image_size
        
        # Camera image dimensions from config
        cam_w, cam_h = self.config["camera_display_size"]  # (640, 360)
        
        # Camera titles
        cam_titles = ['Cam_Front_Left', 'Cam_Front', 'Cam_Front_Right', 'Cam_Rear_Left', 'Cam_Rear', 'Cam_Rear_Right']

        top_imgs, bottom_imgs = [], []
        for i, img in enumerate(camera_imgs):
            if img is None:
                img_resized = np.zeros((cam_h, cam_w, 3), np.uint8)
            else:
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # Resize to camera display size
                img_resized = cv2.resize(img, (cam_w, cam_h))
            
            # Add title text to each camera image
            if i < len(cam_titles):
                img_resized = self.add_text_to_image(img_resized, cam_titles[i])
            
            (top_imgs if i < 3 else bottom_imgs).append(img_resized)

        # Concatenate horizontally: 3 images on top, 3 images on bottom
        top_row = cv2.hconcat(top_imgs)    # cam_h x (3*cam_w) = 360x1920
        bottom_row = cv2.hconcat(bottom_imgs)  # cam_h x (3*cam_w) = 360x1920

        # Stack vertically: top cameras + point cloud + bottom cameras
        return cv2.vconcat([top_row, pointcloud_img, bottom_row])

    def setup_scene(self) -> None:
        """Setup the 3D scene with car model and coordinate frame
        
        Initializes the 3D visualization scene by loading the car model, coordinate frame,
        and the first point cloud frame. Adds all geometries to the visualizer.
        """
        self.car_mesh = self.load_car_model(self.config["car_model"])
        self.coord = self.load_xyz_frame(size=1.0)
        
        # Initialize with first frame
        start = self.config["start_frame"]
        first_data = self.dataset[start]
        base_pcd = self.load_pointcloud_from_array(first_data['velodyne'])
        self.pcd_geom = o3d.geometry.PointCloud(base_pcd)
        
        self.vis.add_geometry(self.pcd_geom)
        self.vis.add_geometry(self.car_mesh)
        self.vis.add_geometry(self.coord)

    def setup_camera(self) -> None:
        """Setup camera view from saved parameters or default view
        
        Attempts to load camera parameters from file. If not found, uses default top-down view.
        """
        cam_params = self.load_camera_parameters(self.config["camera_file"])
        if cam_params:
            ctr = self.vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
            print(f"[Loaded camera] {self.config['camera_file']}")
        else:
            print("[No camera found] Using default view.")
            self.set_view_control_topdown()

    def render_frame(self, frame_idx: int) -> ndarray:
        """Render a single frame and return combined image
        
        Args:
            frame_idx (int): Index of the frame to render from the dataset
            
        Returns:
            ndarray: Combined image containing point cloud and camera views
        """
        data = self.dataset[frame_idx]
        
        # Update point cloud
        pcd = self.load_pointcloud_from_array(data['velodyne'])
        self.pcd_geom.points = pcd.points
        if pcd.has_colors():
            self.pcd_geom.colors = pcd.colors
        
        self.vis.update_geometry(self.pcd_geom)
        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(0.003)
        
        # Capture point cloud view and resize to configured size
        img = self.vis.capture_screen_float_buffer(False)
        img_arr = (255 * np.asarray(img)).astype(np.uint8)
        pcd_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        
        # Resize to configured PCD image size
        pcd_w, pcd_h = self.config["render_window_size"]
        pcd_img = cv2.resize(pcd_img, (pcd_w, pcd_h))
        
        # Add title to PCD image
        pcd_img = self.add_text_to_image(pcd_img, "Velodyne", (20, 40))
        
        # Get camera images
        camera_imgs = []
        cameras = ['image_01', 'image_00', 'image_02', 'image_05', 'image_03', 'image_06']
        for cam_key in cameras:
            img = data.get(cam_key, None)
            camera_imgs.append(img)
        
        # Combine all images
        combined = self.combine_images(pcd_img, camera_imgs)
        
        return combined

    def run(self, mode: str = "render") -> None:
        """Main visualization function with two modes: 'interactive' or 'render'
        
        Args:
            mode (str): Execution mode - either 'interactive' or 'render'
                - 'interactive': Allows manual camera adjustment and saves parameters
                - 'render': Generates video output with pre-configured camera view
                
        Raises:
            RuntimeError: If dataset is empty or video writer initialization fails
        """
        if len(self.dataset) == 0:
            raise RuntimeError("Dataset is empty")
            
        start = self.config["start_frame"]
        end = self.config["end_frame"] if self.config["end_frame"] else len(self.dataset)
        if end > len(self.dataset):
            end = len(self.dataset)

        if mode == "interactive":
            # Interactive mode - adjust view and save camera parameters
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            window_w, window_h = self.config["render_window_size"]
            self.vis.create_window("Adjust view and press 'S' to save camera", window_w, window_h)
            
            self.setup_scene()
            self.set_view_control_topdown()
            
            def save_and_exit(vis):
                self.save_camera_parameters(self.config["camera_file"])
                img = vis.capture_screen_float_buffer(False)
                img_arr = (255 * np.asarray(img)).astype(np.uint8)
                cv2.imwrite("preview_saved.png", cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR))
                print("[Preview saved] preview_saved.png")
                vis.destroy_window()
                return False

            self.vis.register_key_callback(ord("S"), save_and_exit)
            print("Interactive mode: Adjust view and press 'S' to save camera and exit.")
            self.vis.run()
            print("Interactive mode completed.")
            
        elif mode == "render":
            # Render mode - generate video/sequence
            self.vis = o3d.visualization.Visualizer()
            window_w, window_h = self.config["render_window_size"]
            self.vis.create_window("Rendering", window_w, window_h, visible=self.config["show_window"])
            
            self.setup_scene()
            self.setup_camera()
            
            # Setup video writer if needed
            if self.config["save_video"]:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                output_w, output_h = self.config["final_output_size"]
                self.writer = cv2.VideoWriter(
                    self.config["video_save_path"], fourcc, 
                    self.config["fps"], (output_w, output_h)
                )
                if not self.writer.isOpened():
                    raise RuntimeError("VideoWriter initialization failed. Check FFmpeg or path.")

            print(f"[Rendering] frames {start}~{end-1}")

            for i in range(start, end):
                combined = self.render_frame(i)
                
                if self.config["save_video"]:
                    self.writer.write(combined)
                    
                if self.config["show_window"]:
                    cv2.imshow("Combined", combined)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                        break

                print(f"Rendered frame {i+1}/{end}")

            # Cleanup
            if self.writer:
                self.writer.release()
                print(f"[Done] Video saved to {self.config['video_save_path']}")
                
            self.vis.destroy_window()
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for dataset visualization
    
    Returns:
        argparse.Namespace: Parsed arguments containing all configuration parameters
    """
    parser = argparse.ArgumentParser(
        description='Visualize Hella Dataset with point clouds and camera images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset parameters
    parser.add_argument('--dataset_dir', type=str, 
                        default='/p/data1/nxtaim/proprietary/hella/HellaDataset/',
                        help='Path to the Hella dataset directory')
    parser.add_argument('--seq_list', type=str, nargs='+',
                        default=['seq15'],
                        help='List of sequence names to load (e.g., seq15 seq16)')
    parser.add_argument('--sensors', type=str, nargs='+',
                        default=['velodyne', 'oxts', 'cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6'],
                        help='List of sensors to load from the dataset')
    
    # Visualization model parameters
    parser.add_argument('--car_model', type=str,
                        default='assets/Car.obj',
                        help='Path to the car 3D model file (.obj format)')
    parser.add_argument('--camera_file', type=str,
                        default='assets/camera.json',
                        help='Path to save/load camera view parameters')
    
    # Output parameters
    parser.add_argument('--video_save_path', type=str,
                        default='images/output.mp4',
                        help='Output video file path')
    parser.add_argument('--fps', type=int,
                        default=20,
                        help='Frames per second for output video')
    
    # Mode parameters
    parser.add_argument('--interactive', action='store_true',
                        default=False,
                        help='Run in interactive mode to adjust camera view (press S to save)')
    parser.add_argument('--show_window', action='store_true',
                        default=True,
                        help='Show visualization window during rendering')
    parser.add_argument('--save_video', action='store_true',
                        default=True,
                        help='Save output to video file')
    
    # Frame range parameters
    parser.add_argument('--start_frame', type=int,
                        default=0,
                        help='Starting frame index for rendering')
    parser.add_argument('--end_frame', type=int,
                        default=300,
                        help='Ending frame index for rendering (0 means all frames)')
    
    # Image size parameters (all in width x height format)
    parser.add_argument('--rgb_image_size', type=int, nargs=2,
                        default=[1920, 1080],
                        help='Original RGB camera image resolution (width height)')
    parser.add_argument('--camera_display_size', type=int, nargs=2,
                        default=[640, 360],
                        help='Resized camera image size for display (width height)')
    parser.add_argument('--final_output_size', type=int, nargs=2,
                        default=[1920, 1680],
                        help='Final combined output video resolution (width height)')
    parser.add_argument('--render_window_size', type=int, nargs=2,
                        default=[1920, 960],
                        help='Open3D rendering window size (width height)')
    
    args = parser.parse_args()
    return args


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse arguments to configuration dictionary
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
    
    Returns:
        Dict[str, Any]: Configuration dictionary for visualizer with properly typed values
    """
    config = {
        "car_model": args.car_model,
        "camera_file": args.camera_file,
        "video_save_path": args.video_save_path,
        "fps": args.fps,
        "interactive": args.interactive,
        "show_window": args.show_window,
        "save_video": args.save_video,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        # Image size configurations (convert list to tuple for consistency)
        "rgb_image_size": tuple(args.rgb_image_size),
        "camera_display_size": tuple(args.camera_display_size),
        "final_output_size": tuple(args.final_output_size),
        "render_window_size": tuple(args.render_window_size),
    }
    return config


# ================= Main Logic =================
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Print configuration
    print("=" * 60)
    print("Hella Dataset Visualizer Configuration")
    print("=" * 60)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Sequences: {args.seq_list}")
    print(f"Sensors: {args.sensors}")
    print(f"Frame range: {args.start_frame} to {args.end_frame}")
    print(f"Mode: {'Interactive' if args.interactive else 'Render'}")
    print(f"Output video: {args.video_save_path}")
    print(f"FPS: {args.fps}")
    print(f"Show window: {args.show_window}")
    print(f"Save video: {args.save_video}")
    print(f"Render window size: {args.render_window_size}")
    print(f"Camera display size: {args.camera_display_size}")
    print(f"Final output size: {args.final_output_size}")
    print("=" * 60)
    
    # Initialize dataset
    print("\nLoading dataset...")
    dataset = HellaDataset(
        dataset_dir=args.dataset_dir,
        seq_list=args.seq_list,
        sensors=args.sensors
    )
    print(f"Dataset loaded: {len(dataset)} frames")

    # Convert args to config dictionary
    config = args_to_config(args)

    # Create and run visualizer
    print("\nInitializing visualizer...")
    visualizer = HellaDatasetVisualizer(dataset, config)
    
    if config["interactive"]:
        print("\nStarting interactive mode...")
        print("Instructions:")
        print("  - Use mouse to rotate, zoom, and pan the view")
        print("  - Press 'S' to save camera parameters and exit")
        visualizer.run(mode="interactive")
    else:
        print("\nStarting render mode...")
        visualizer.run(mode="render")
    
    print("\n" + "=" * 60)
    print("Visualization completed!")
    print("=" * 60)

