# Binary-Image-Autoencoder
**Binary Image Autoencoder** is a neural network model that encodes binary data into images and decodes images back into binary data.

This project uses advanced machine learning techniques to transform binary data into visual representations (images) and then reconstructs these images back into their original binary form. It utilizes convolutional layers to extract meaningful features from images and fully connected layers to encode and decode data.

--**>Message Processor**: Processes input noise to prepare it for concatenation with the image data.

-->**Encoder**: Extracts features from concatenated noise and image data.

-->**Decoder**: Reconstructs binary code and images from encoded features.

-->**Loss Functions**: Combines binary cross-entropy loss, mean squared error loss, and LPIPS perceptual loss for training.

-->**Training Scripts**: Includes data loading, preprocessing, training loop, and checkpoint saving.

-->**Visualization Tools**: Plots and saves original and reconstructed images at regular intervals during training.
