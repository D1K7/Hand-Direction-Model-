# Hand-Direction-Model-


This project implements a custom Convolutional Neural Network (CNN) built with TensorFlow to interpret hand-pointing gestures via a webcam. The system is designed to eventually control a drone (e.g., DJI Tello) by translating visual gestures into directional commands.

Features
CNN Architecture: A multi-layer CNN trained from scratch on a local dataset.

Grayscale Image Processing: Converts RGB input to 1-channel grayscale to eliminate interference from skin tone, clothing color, and background noise.

Temporal Prediction Smoothing: Uses a collections.deque buffer to "vote" on the last 10 frames, preventing jerky drone movements from flickering predictions.

5-Class Control System: * UP, DOWN, LEFT, RIGHT, and IDLE (Background/No Command).
