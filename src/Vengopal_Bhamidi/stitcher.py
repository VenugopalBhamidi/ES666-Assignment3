import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

class PanaromaStitcher:
    def detect_and_compute(self, images):
        sift = cv2.SIFT_create()
        keypoints = []
        descriptors = []
        for img in images:
            kp, desc = sift.detectAndCompute(img, None)
            keypoints.append(kp)
            descriptors.append(desc)
        return keypoints, descriptors

    def compute_pairwise_homography(self, images, keypoints, descriptors):
        homographies = [np.eye(3)]
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

        for i in range(len(images) - 1):
            keypoints1 = keypoints[i]
            keypoints2 = keypoints[i + 1]
            matches = matcher.knnMatch(descriptors[i], descriptors[i + 1], k=2)

            # Lowe's ratio test
            good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]

            pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            H, mask = self.compute_homography_ransac(pts1, pts2, threshold=5.0, max_iterations=1500)
            homographies.append(H)

        return homographies


    def compute_total_homographies(self, homographies, target_idx=2):
        total_homographies = []
        H_target = np.eye(3)

        for i, H in enumerate(homographies):
            if i < target_idx:
                H_total = np.eye(3)
                for j in range(i, target_idx):
                    H_total = np.dot(homographies[j + 1], H_total)
                total_homographies.append(H_total)
            elif i > target_idx:
                H_total = np.eye(3)
                for j in range(i, target_idx, -1):
                    H_total = np.dot(np.linalg.inv(homographies[j]), H_total)
                total_homographies.append(H_total)
            else:
                total_homographies.append(H_target)

        return total_homographies

    def compute_homography_ransac(self, pts1, pts2, threshold=5.0, max_iterations=1000):
        best_H = None
        max_inliers = 0
        best_inliers_mask = None
        np.random.seed(4)
        pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))

        for _ in range(max_iterations):
            indices = np.random.choice(len(pts1), 4, replace=False)
            subset_pts1 = pts1[indices]
            subset_pts2 = pts2[indices]
            H = self.find_homography(subset_pts1, subset_pts2)

            if H is None:
                continue

            projected_pts2_h = (pts1_h @ H.T)
            projected_pts2_h /= projected_pts2_h[:, 2:3]
            distances = np.linalg.norm(pts2 - projected_pts2_h[:, :2], axis=1)

            inliers_mask = distances < threshold
            num_inliers = np.sum(inliers_mask)

            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_H = H
                best_inliers_mask = inliers_mask

        return best_H, best_inliers_mask

    def find_homography(self, src_pts, dst_pts):
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i]
            xp, yp = dst_pts[i]
            A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
            A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])

        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        return H / H[2, 2] if H[2, 2] != 0 else None

    def warp_image(self, images, homographies):

        canvas_height = max(img.shape[0] for img in images) * 4
        canvas_width = sum(img.shape[1] for img in images)
        translation_matrix = np.array([[1, 0, canvas_width // 4], [0, 1, canvas_height // 4], [0, 0, 1]], dtype=np.float32)
        panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        for i, H in enumerate(homographies):
            H_translated = np.dot(translation_matrix, H)
            warped_image = self.warp_perspective(images[i], H_translated, (canvas_height, canvas_width))
            panorama = np.maximum(panorama, warped_image)
        #Gray scaling it. (Just for reference code manual)
        gray_panorama = np.dot(panorama[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        binary_mask = (gray_panorama > 1).astype(np.uint8) * 255
        coords = np.column_stack(np.where(binary_mask > 0))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
        else:
            y_min, x_min, y_max, x_max = 0, 0, 0, 0

        cropped_panorama = panorama[y_min:y_max + 1, x_min:x_max + 1]

        return cropped_panorama, homographies


    def warp_perspective(self, image, H, output_shape):
        h_out, w_out = output_shape
        warped_image = np.zeros((h_out, w_out, image.shape[2]), dtype=image.dtype)
        H_inv = np.linalg.inv(H)

        x_coords, y_coords = np.meshgrid(np.arange(w_out), np.arange(h_out))
        ones = np.ones_like(x_coords)
        destination_coords = np.stack((x_coords, y_coords, ones), axis=-1).reshape(-1, 3)

        src_coords = (H_inv @ destination_coords.T).T
        src_coords /= src_coords[:, 2].reshape(-1, 1)
        src_x = src_coords[:, 0]
        src_y = src_coords[:, 1]

        valid_mask = (0 <= src_x) & (src_x < image.shape[1]) & (0 <= src_y) & (src_y < image.shape[0])
        src_x_clipped = np.clip(src_x, 0, image.shape[1] - 1)
        src_y_clipped = np.clip(src_y, 0, image.shape[0] - 1)

        x0 = np.floor(src_x_clipped).astype(np.int32) #to get relevant information only.
        x1 = np.clip(x0 + 1, 0, image.shape[1] - 1)
        y0 = np.floor(src_y_clipped).astype(np.int32)
        y1 = np.clip(y0 + 1, 0, image.shape[0] - 1)
        dx = src_x_clipped - x0
        dy = src_y_clipped - y0

        for c in range(image.shape[2]):
            top_left = image[y0, x0, c] * (1 - dx) + image[y0, x1, c] * dx
            bottom_left = image[y1, x0, c] * (1 - dx) + image[y1, x1, c] * dx
            warped_channel = top_left * (1 - dy) + bottom_left * dy
            warped_image[..., c].flat[valid_mask] = warped_channel[valid_mask]
        return warped_image

    def make_panaroma_for_images_in(self, path):
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(all_images)))
        images = []
        for img in all_images:
            image = cv2.imread(img)
            if image is not None and image.size != 0:
                images.append(image)
            else:
                print(f"Warning: Could not load or image is empty: {img}")

        keypoints, descriptors = self.detect_and_compute(images)
        pairwise_homographies = self.compute_pairwise_homography(images, keypoints, descriptors)
        total_homographies = self.compute_total_homographies(pairwise_homographies, target_idx=len(images) // 2)
        stitched_img, homographies = self.warp_image(images, total_homographies)
        stitched_img = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB)
        return stitched_img, homographies
