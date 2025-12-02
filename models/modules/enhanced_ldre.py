#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: enhanced_ldre.py
====================================
Enhanced LDRE (Keypoint-guided Local Dropout for Regularization Enhancement) Module

Purpose:
- Provide keypoint-guided region dropping for improved regularization
- Enhance model robustness by focusing on semantically important regions
- Support adaptive dropout based on keypoint importance

Features:
1. Keypoint-Guided: Utilizes keypoint information to guide dropout regions
2. Adaptive Dropout: Dynamically adjusts dropout regions based on input
3. Performance Optimized: Efficient implementation for training speed
4. Configurable: Adjustable grid size and dropout ratio
5. Integration Ready: Seamlessly integrates with PetNet architecture
"""
import numpy as np
import random

class EnhancedLDRETransform:
    def __init__(self, grid_size: int = 16, top_k_ratio: float = 0.2, prob: float = 0.5):
        """
        Enhanced LDRE with keypoint-based region dropping

        Args:
            grid_size: Grid size for creating score map (16x16)
            top_k_ratio: Ratio of top grid cells to drop (e.g., 0.2 = top 20%)
            prob: Probability of applying LDRE
        """
        self.grid_size = grid_size
        self.top_k_ratio = top_k_ratio
        self.prob = prob

    def _create_score_map(self, image_shape, keypoints):
        """
        Create score map based on keypoint locations

        Args:
            image_shape: Tuple of (height, width)
            keypoints: List of keypoint dictionaries with x, y, score

        Returns:
            score_map: (grid_size, grid_size) float array
        """
        h, w = image_shape[:2]
        cell_w = w / self.grid_size
        cell_h = h / self.grid_size

        # Initialize score map
        score_map = np.zeros((self.grid_size, self.grid_size))

        for kp in keypoints:
            if 'x' not in kp or 'y' not in kp:
                continue

            x = kp['x']
            y = kp['y']
            score = kp.get('score', 1.0)

            grid_x = min(int(x // cell_w), self.grid_size - 1)
            grid_y = min(int(y // cell_h), self.grid_size - 1)
            print(f"Img Shape: {image_shape}, KP: ({x}, {y}) -> Grid: ({grid_x}, {grid_y})")
            score_map[grid_y, grid_x] += score

        return score_map

    def __call__(self, image, keypoints=None):
        """
        Apply LDRE augmentation with keypoint-guided region dropping

        Args:
            image: Raw image array (H, W, C)
            keypoints: List of keypoint dictionaries with x, y, score

        Returns:
            Processed image array
        """
        if random.random() > self.prob or keypoints is None or not keypoints:
            return image

        h, w, c = image.shape
        processed_image = image.copy()
        score_map = self._create_score_map((h, w), keypoints)

        # Get top-k grid cells
        flat_scores = score_map.flatten()
        k = int(len(flat_scores) * self.top_k_ratio)
        if k == 0:
            k = 1

        non_zero_mask = flat_scores > 0
        if np.any(non_zero_mask):
            top_indices = np.argpartition(flat_scores[non_zero_mask], -k)[-k:]
            flat_indices = np.where(non_zero_mask)[0][top_indices]
        else:
            flat_indices = np.random.choice(
                len(flat_scores),
                size=min(k, len(flat_scores)),
                replace=False
            )

        grid_y, grid_x = np.unravel_index(flat_indices, score_map.shape)
        cell_w = w // self.grid_size
        cell_h = h // self.grid_size

        for y, x in zip(grid_y, grid_x):
            x1 = x * cell_w
            y1 = y * cell_h
            x2 = min(x1 + cell_w, w)
            y2 = min(y1 + cell_h, h)

            if x2 > x1 and y2 > y1:
                if random.random() > 0.5:
                    # Mean filling
                    region_mean = np.mean(image[y1:y2, x1:x2], axis=(0, 1))
                    processed_image[y1:y2, x1:x2] = region_mean
                else:
                    # Noise addition
                    noise = np.random.normal(0, 25, (y2-y1, x2-x1, c)).astype(np.uint8)
                    processed_image[y1:y2, x1:x2] = np.clip(
                        image[y1:y2, x1:x2].astype(np.int16) + noise, 0, 255
                    ).astype(np.uint8)

        return processed_image