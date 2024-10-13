# Glaucoma-Eye-Semantic-Segmentation-UNET

![image](https://github.com/user-attachments/assets/157a810d-0196-4766-9769-eb9edb8e0572)

Glaucoma Detection
Introduction

Glaucoma is one of the leading causes of blindness worldwide, characterized by damage to the optic nerve. This condition often develops slowly over time, impairing vision and leading to blindness if left untreated. Glaucoma poses a significant public health challenge due to its prevalence, lifelong management requirements, and the absence of early symptoms, resulting in late diagnoses and vision loss.
Diagnosis Challenges

One of the primary challenges in glaucoma diagnosis lies in its asymptomatic nature during the early stages. Patients may not experience noticeable symptoms until significant vision loss occurs, making early detection crucial. Traditional diagnostic methods, such as intraocular pressure measurement and optic nerve examination, may not always detect glaucoma in its early stages or accurately assess disease progression. This limitation underscores the need for precise and reliable diagnostic tools to enable timely intervention and improve patient outcomes.
Importance of the Project

Accurate and early diagnosis is critical in managing glaucoma. Early detection allows for interventions—medication, laser therapy, or surgery—to slow or halt disease progression and preserve vision. Effective monitoring of disease progression is also essential for adjusting treatment plans and optimizing patient care.

This project focuses on developing a robust segmentation model for glaucoma eye images. The segmentation of key structures within the eye, such as the optic disc and cup, is vital for diagnosing and monitoring glaucoma. By accurately delineating these structures, the proposed model aims to provide clinicians with insights into disease severity and progression, facilitating early detection and improved patient outcomes.

Methodology

    Loading Images and Masks:
        The load_images_and_masks function loads images and their corresponding masks from specified directories using glob.glob.
        Each image is read using OpenCV and converted from BGR to RGB color space, resized to 128x128 pixels, and normalized to the range [0, 1].
        Masks are read as grayscale images, resized to 128x128 pixels using nearest-neighbor interpolation, normalized to [0, 1], and expanded along the channel dimension to match the model's input shape.
        The function returns two lists: images (preprocessed images) and masks (preprocessed masks).

    Data Preprocessing and Splitting:
        Masks are encoded using LabelEncoder and transformed into categorical format using one-hot encoding.
        The images and one-hot encoded masks are split into training (80%) and testing sets (20%) using train_test_split.

    U-Net Model Architecture:
        The U-Net architecture consists of encoder-decoder structures with skip connections.
        The model is compiled using the Adam optimizer and categorical cross-entropy loss, with accuracy, recall, and precision as evaluation metrics.

    Model Training:
        The compiled U-Net model is trained using the fit method, adjusting weights to minimize categorical cross-entropy loss.
        Validation data monitors performance on unseen data to prevent overfitting.

    Display:
        Original and segmented images are displayed using the Matplotlib library to visualize results.

    CDR Calculation:
        Counted the number of pixels for the optical cup and optical disk from the predicted mask.
        Divided both and multiplied by 100 to get CDR Percentage.

Technical Details

    Input Layer:
        Shape: (128, 128, 3) - Accepts RGB images of size 128x128 pixels.

    Encoder Blocks (Contracting Path):
        Four encoder blocks, each with:
            Two convolutional layers (3x3 filter size).
            ReLU activation and batch normalization after each convolution.
        Filter sizes: 32, 64, 128, 256.

    Max Pooling Layers:
        Used after each convolutional block in the encoder with a pooling size of (2, 2).

    Bridge:
        Single bridge layer with a convolutional block containing 512 filters.

    Decoder Blocks (Expanding Path):
        Four decoder blocks mirroring encoder blocks with:
            Transposed convolutional layers for upsampling.
            Concatenation with corresponding skip connections.
            Two convolutional layers (3x3 filter size).
        Filter sizes: 256, 128, 64, 32.

    Output Layer:
        Single convolutional layer with softmax activation producing a segmentation map with three channels (for three classes).

    Activation Function:
        ReLU used throughout, softmax in the output layer.

    Loss Function:
        Categorical Cross-Entropy for multi-class classification.

    Optimizer:
        Adam optimizer.

    Metrics:
        Accuracy, Recall, and Precision during training.

Training Process Details

    Optimizer Choice:
        Adam optimizer dynamically adjusts the learning rate, suitable for high-dimensional parameter spaces.

    Learning Rate Schedule:
        Default learning rate of 0.001, adjusting automatically based on training progress and gradients.

    Early Stopping Callback:
        Early stopping monitors validation loss. Training stops if the validation loss doesn't improve for a set number of epochs (patience of 5), restoring the model's best weights.

Results

    Achieved an Accuracy of 99.5896% on the test dataset.
    Achieved a precision of 99.59% on validation data.
    Achieved a recall of 99.59% on validation data.

Conclusion

The development of a U-Net architecture for segmenting the optical disk and cup from retinal images represents a significant advancement in automated glaucoma detection. By leveraging deep learning techniques, the model achieves robust performance in accurately identifying critical features essential for diagnosing glaucoma.

This U-Net model is valuable for large-scale screening programs, particularly in underserved regions, facilitating rapid and reliable analysis of retinal images. By democratizing glaucoma detection, the model aims to improve patient outcomes, preserve vision, and combat the global burden of glaucoma.
