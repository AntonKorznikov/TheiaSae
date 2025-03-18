import os
import numpy as np
from PIL import Image
from typing import Dict, List, Optional
import skimage
import scipy
from ..base_dataset import BaseDataset, register_dataset
from ..common_utils import download_and_extract

def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
    """Interpolate sparse depth map using colorization method."""
    imgIsNoise = imgDepthInput == 0
    maxImgAbsDepth = np.max(imgDepthInput)
    imgDepth = imgDepthInput / maxImgAbsDepth
    imgDepth[imgDepth > 1] = 1
    (H, W) = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape((W, H)).transpose()
    knownValMask = (imgIsNoise == False).astype(int)
    grayImg = skimage.color.rgb2gray(imgRgb)
    winRad = 1
    len_ = 0
    absImgNdx = 0
    len_window = (2 * winRad + 1) ** 2
    len_zeros = numPix * len_window

    cols = np.zeros(len_zeros) - 1
    rows = np.zeros(len_zeros) - 1
    vals = np.zeros(len_zeros) - 1
    gvals = np.zeros(len_window) - 1

    for j in range(W):
        for i in range(H):
            nWin = 0
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue

                    rows[len_] = absImgNdx
                    cols[len_] = indsM[ii, jj]
                    gvals[nWin] = grayImg[ii, jj]

                    len_ = len_ + 1
                    nWin = nWin + 1

            curVal = grayImg[i, j]
            gvals[nWin] = curVal
            c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin+ 1])) ** 2)

            csig = c_var * 0.6
            mgv = np.min((gvals[:nWin] - curVal) ** 2)
            if csig < -mgv / np.log(0.01):
                csig = -mgv / np.log(0.01)

            if csig < 2e-06:
                csig = 2e-06

            gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
            gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
            vals[len_ - nWin:len_] = -gvals[:nWin]

            rows[len_] = absImgNdx
            cols[len_] = absImgNdx
            vals[len_] = 1

            len_ = len_ + 1
            absImgNdx = absImgNdx + 1

    vals = vals[:len_]
    cols = cols[:len_]
    rows = rows[:len_]
    A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    rows = np.arange(0, numPix)
    cols = np.arange(0, numPix)
    vals = (knownValMask * alpha).transpose().reshape(numPix)
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    A = A + G
    b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

    new_vals = scipy.sparse.linalg.spsolve(A, b)
    new_vals = np.reshape(new_vals, (H, W), 'F')

    denoisedDepthImg = new_vals * maxImgAbsDepth
    
    output = denoisedDepthImg.reshape((H, W)).astype('float32')
    output = np.multiply(output, (1-knownValMask)) + imgDepthInput
    
    return output

def kitti_benchmark_crop(input_img):
    """
    Crop PIL image to KITTI benchmark size
    
    Args:
        input_img (PIL.Image): Input image to be cropped.

    Returns:
        PIL.Image: Cropped image.
    """
    KB_CROP_HEIGHT = 352
    KB_CROP_WIDTH = 1216
    
    width, height = input_img.size
    top_margin = int(height - KB_CROP_HEIGHT)
    left_margin = int((width - KB_CROP_WIDTH) / 2)
    
    # Crop the image
    crop_box = (left_margin, top_margin, left_margin + KB_CROP_WIDTH, top_margin + KB_CROP_HEIGHT)
    cropped_img = input_img.crop(crop_box)
    
    return cropped_img


@register_dataset('kitti')
class KITTIDataset(BaseDataset):
    """KITTI depth dataset."""
    min_depth=1e-5
    max_depth=80.0
    
    def _traverse_directory(self) -> List[Dict[str, str]]:
        """Find all matching RGB-D pairs in the dataset.

        Returns:
            List[Dict[str, str]]: List of dictionaries containing paired 'rgb' and 'depth' file paths
        """
        data_pairs = []

        # Iterate through drive dates
        for drive_date in os.listdir(self.root_dir):
            if drive_date.endswith("sync"):
                continue

            date_path = os.path.join(self.root_dir, drive_date)

            # Iterate through drive numbers
            for drive_num in os.listdir(date_path):
                rgb_dir = os.path.join(date_path, drive_num, "image_02", "data")
                depth_dir = os.path.join(
                    self.root_dir, drive_num, "proj_depth/groundtruth/image_02"
                )

                # Skip if RGB directory doesn't exist
                if not os.path.exists(rgb_dir):
                    continue

                # Get all RGB images
                for img_name in os.listdir(rgb_dir):
                    if not img_name.endswith(".png"):
                        continue

                    rgb_path = os.path.join(rgb_dir, img_name)
                    depth_path = os.path.join(depth_dir, img_name)

                    # Verify both files exist
                    try:
                        # Quick verification that depth file can be opened
                        Image.open(depth_path)

                        # Add valid pair to dataset
                        data_pairs.append({"rgb": rgb_path, "depth": depth_path})
                    except (FileNotFoundError, IOError):
                        continue

        return sorted(data_pairs, key=lambda x: x["rgb"])

    def _load_rgb_image(self, path: str) -> Image.Image:
        """Load RGB image with KITTI-specific cropping.
        
        Args:
            path: Path to RGB image
            
        Returns:
            Cropped RGB image as PIL Image
        """
        rgb_img = Image.open(path).convert('RGB')
        return kitti_benchmark_crop(rgb_img)
    
    def _load_depth(self, path: str) -> Image.Image:
        """Load KITTI depth map with special processing.
        
        KITTI depth maps are uint16 PNGs with depth values
        encoded in the actual pixel values.
        
        Args:
            path: Path to depth map file
            
        Returns:
            Depth map as PIL Image
        """
        depth_img = Image.open(path)
        
        # Apply KITTI benchmark crop
        depth_img_cropped = kitti_benchmark_crop(depth_img)
        
        # For KITTI, we need to convert the depth values
        # We'll return the processed image
        # The actual scaling (divide by 256.0) can be done during evaluation
        return depth_img_cropped
    
    def _get_valid_mask(self, depth: Image.Image) -> np.ndarray:
        """Get valid mask for depth map with KITTI-specific crop.
        
        Args:
            depth: Depth map as PIL Image
            
        Returns:
            Valid mask as numpy array (boolean)
        """
        # Convert PIL Image to numpy array and scale appropriately
        depth_np = np.array(depth).astype(np.float32) / 256.0
        
        # Create base valid mask
        valid_mask = np.logical_and(
            (depth_np > self.min_depth), (depth_np < self.max_depth)
        )
        
        # Apply "eigen" evaluation crop if needed
        h, w = valid_mask.shape
        eval_mask = np.zeros_like(valid_mask, dtype=bool)
        
        # "eigen" crop
        eval_mask[
            int(0.3324324 * h) : int(0.91351351 * h),
            int(0.0359477 * w) : int(0.96405229 * w),
        ] = True
        
        # Combine masks
        valid_mask = np.logical_and(valid_mask, eval_mask)
        
        return valid_mask
    
    def _download(self):
        """Download and extract KITTI dataset if not already present."""
        # Check if data already exists
        if os.path.exists(self.root_dir) and len(os.listdir(self.root_dir)) > 0:
            print("KITTI dataset already exists.")
            return

        url = "https://huggingface.co/datasets/guangkaixu/genpercept_datasets_eval/resolve/main/eval_kitti_genpercept.tar.gz?download=true"
        print("Downloading KITTI dataset...")
        download_and_extract(
            url=url,
            download_dir=os.path.dirname(self.root_dir),
            extract_dir=os.path.dirname(self.root_dir),
        )