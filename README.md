<h1><strong>AWS AI/ML Scholarship Program Fall 2024 with Udacity</strong></h1>

<p><strong>Author:</strong> Seungmi Kim (kimsm6397@gmail.com)<br></p>
<p><strong>Repository:</strong> <code>awsAIMLscholarship-fa24<br></code></p>
<p><strong>Last Updated:</strong> February 4, 2025</p>

<p>
Welcome to the <code>awsAIMLscholarship-fa24</code> repository. This repository is dedicated to projects as part of the <strong>AI Programming with Python Nanodegree</strong> from Udacity, awarded through the <strong>AWS AI & ML Scholarship Program</strong>. This Nanodegree focuses on building essential AI and Python programming skills, with an opportunity to advance into the <strong>AWS ML Fundamentals Nanodegree</strong> for selected students.
</p>
<p><strong>Program Highlights</strong></p>
<ul>
    <li>Selected as part of a global initiative awarding 2,000 scholarships.</li>
    <li>Focus on foundational AI and Python skills with future access to specialized ML content.</li>
    <li>Opportunity to work on real-world projects that enhance practical understanding and programming skills in AI and machine learning.</li>
</ul>
<p><strong>List of Projects</strong></p>
<ul>
    <li><a href="#project1">Project 1: City Dog Show Image Classification</a></li>
    <li><a href="#project2">Project 2: Create Your Own Image Classifier</a></li>
    <li><a href="#project3">Project 3: Bike Sharing Data EDA and Business Recommendations</a></li>
</ul>

<br><br>
<h2 id="project1"><strong>Project 1: City Dog Show Image Classification</strong> </h2> 

<p>
The <strong>City Dog Show Image Classification</strong> project provides hands-on experience in computer vision through a practical application: automated dog breed identification. This implementation demonstrates fundamental deep learning techniques by utilizing convolutional neural networks (CNNs) for image classification. The project explores three widely-adopted CNN architectures—ResNet, AlexNet, and VGG—to classify dog breeds with high accuracy.
</p>
<p><strong>Project Goal:</strong> This project aims to develop practical Python implementation skills by utilizing pre-trained CNN for accurate dog breed classification. The project utilizes a custom classifier function (located in <code>classifier.py</code>) that includes established CNN architectures pre-trained on the ImageNet dataset to perform robust image classification tasks.</p>

<p><strong>List of Objectives</strong></p>
<ul>
    <li>Correctly identify images of dogs (even if the breed is misclassified) and differentiate them from non-dog images.</li>
    <li>Classify the breed of dog for correctly identified dog images.</li>
    <li>Evaluate the performance of different CNN architectures (ResNet, AlexNet, VGG) to determine which one best meets the objectives.</li>
    <li>Compare the computation time for each algorithm to achieve a balance between accuracy and runtime.</li>
</ul>

<p><strong>Project Structure</strong></p>
<ul>
    <li><code>check_images.py</code> is the main Python script, handling essential functions including image processing, performance timing, command-line argument parsing, classification comparison, and results generation. This main script orchestrates the entire classification workflow.</li>
    <li><code>classifier.py</code> contains the pre-trained CNN architectures (ResNet, AlexNet, and VGG), providing a unified interface for image classification. This script is to aid consistent breed identification across different neural network architectures.</li>
    <li>Supporting files include <code>dognames.txt</code>, which maintains the comprehensive list of recognized dog breeds, and the <code>pet_images/</code> directory that stores the image dataset for classification and analysis.</li>
</ul>

<p><strong>Future Improvements:</strong> The project roadmap includes potential improvements such as model fine-tuning for domain-specific accuracy, expanding the breed recognition database, and implementing real-time classification capabilities.</p>

<p><strong>Prerequisites</strong></p>
<ul>
    <li><strong>Python Environment:</strong> Python 3.x installation required</li>
    <li><strong>Dependencies:</strong> Deep learning frameworks including PyTorch, torchvision, and NumPy (full list in <code>requirements.txt</code>)</li>
    <li><strong>Dataset:</strong> Valid image files must be present in the pet_images/ directory</li>
</ul>

<p><strong>Execution Guide</strong></p>
<ol type="i">
    <li><strong>Setup</strong>: Install all required packages.<br>
    <pre><code>pip install -r requirements.txt</code></pre></li>
    <li><strong>Run Individual Models</strong>: Execute each model with specific configurations.<br>
    <pre><code>python check_images.py --dir pet_images/ --arch resnet --dogfile dognames.txt > resnet_pet-images.txt
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt > alexnet_pet-images.txt
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt > vgg_pet-images.txt</code></pre></li>
    <li><strong>Batch Execution</strong>: Alternatively, run all models sequentially.<br>
<pre><code>sh run_models_batch.sh</code></pre></li>
</ol>
 
<p><strong>Output:</strong> The program generates comprehensive performance metrics, including classification accuracy rates, breed identification precision, and computational efficiency measurements for each neural network architecture. These results will be printed and saved as specified.</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/d3dc6824-1daf-4f12-bbec-e48de3fe4b7f" alt="AWSWinter24-Project1-small">
</p>
<p align="center">
  <strong><em>Project 1 Completion Badge</em></strong>
</p>





<br><br>
<h2 id="project2"><strong>Project 2: Create Your Own Image Classifier</strong></h2> 

<p>
The <strong>Create Your Own Image Classifier</strong> project focuses on developing a flower image classification model using deep learning. This project utilizes <strong>transfer learning</strong> with pre-trained convolutional neural networks (CNNs), specifically <strong>VGG16</strong>, to classify images of flowers. The main goal is to build an end-to-end AI application that takes an image as input and returns a prediction of the flower species.
</p>

<p><strong>Project Goal:</strong> This project applies <strong>deep learning and PyTorch</strong> to train a custom image classifier on the <strong>102-category Oxford Flower dataset</strong>. The trained model can recognize and classify flower images with high accuracy and is further integrated into a <strong>command-line application</strong> for real-world usability.</p>

<p><strong>List of Objectives</strong></p>
<ul>
    <li>Load and preprocess image datasets (train, validation, and test sets).</li>
    <li>Implement transfer learning using <strong>pre-trained CNNs (VGG16)</strong>.</li>
    <li>Fine-tune a custom <strong>fully connected classifier</strong> with <strong>dropout and ReLU activations</strong>.</li>
    <li>Train the classifier using <strong>cross-entropy loss</strong> and <strong>backpropagation</strong> with GPU acceleration.</li>
    <li>Validate the model’s performance and achieve <strong>at least 70% accuracy</strong> on the test set.</li>
    <li>Save and load model checkpoints to <strong>avoid retraining from scratch</strong>.</li>
    <li>Build a <strong>command-line application</strong> that allows users to classify images from the terminal.</li>
</ul>

<p><strong>Training Performance Report</strong></p>
<ul>
    <li><strong>Model:</strong> VGG16 (Pre-trained)</li>
    <li><strong>Training Duration:</strong> 5 epochs</li>
    <li><strong>Final Training Loss:</strong> 1.451</li>
    <li><strong>Final Validation Loss:</strong> 0.795</li>
    <li><strong>Final Validation Accuracy:</strong> 78.9%</li>
    <li><strong>Test Accuracy:</strong> 76.0%</li>
</ul>

<p><strong>Observations:</strong></p>
<ul>
    <li>Training loss <strong>steadily decreases</strong>, indicating effective learning.</li>
    <li>Validation accuracy <strong>consistently increases</strong>, reaching <strong>~79%</strong>, exceeding the target of 70%+.</li>
    <li>The model shows <strong>no significant overfitting</strong>, as validation loss also decreases.</li>
    <li>On test data, the model achieved a <strong>test accuracy of 76.0%</strong>, demonstrating strong generalization.</li>
</ul>

<p><strong>Project Structure</strong></p>
<p>
    The model and dataset can be used through both <strong>command-line execution</strong> and a <strong>Jupyter Notebook</strong>:
</p>
<ul>
    <li><code>train.py</code>: Trains a new deep learning model on a dataset and saves it as a checkpoint.</li>
    <li><code>predict.py</code>: Loads a trained model and predicts the class of a given image.</li>
    <li><code>model_utils.py</code>: Contains reusable functions for model creation, training, and checkpoint saving/loading.</li>
    <li><code>data_utils.py</code>: Handles data loading and preprocessing for training and inference.</li>
    <li><code>process_image.py</code>: Preprocesses input images for prediction, including resizing, cropping, and normalization.</li>
    <li><code>image_classifier_project.ipynb</code>: A Jupyter Notebook version of the workflow that allows interactive training and inference.</li>
    <li><code>cat_to_name.json</code>: Maps class indices to actual flower names for human-readable predictions.</li>
</ul>

<p><strong>Future Improvements:</strong> Future developments might include:</p>
<ul>
    <li>Improving model accuracy with <strong>data augmentation and hyperparameter tuning</strong>.</li>
    <li>Implementing <strong>real-time image classification</strong> via a Flask or FastAPI web service.</li>
    <li>Extending the model to <strong>support custom datasets</strong> beyond flowers.</li>
    <li>Deploying the classifier as a <strong>mobile app or web application</strong> for broader accessibility.</li>
</ul>

<p><strong>Prerequisites</strong></p>
<ul>
    <li><strong>Python Environment:</strong> Python 3.x installation required.</li>
    <li><strong>Required Libraries:</strong> PyTorch, torchvision, PIL, numpy, argparse (install via <code>requirements.txt</code>).</li>
    <li><strong>Dataset:</strong> The <strong>Oxford 102 Flowers dataset</strong> (download and organize into <code>train</code>, <code>valid</code>, and <code>test</code> directories).</li>
</ul>

<p><strong>Execution Guide</strong></p>
<ol type="i">
    <li><strong>Install dependencies</strong>:<br>
    <pre><code>pip install -r requirements.txt</code></pre></li>
    <li><strong>Train a new model</strong> (example using VGG16):<br>
    <pre><code>python train.py flowers --arch vgg16 --learning_rate 0.003 --hidden_units 512 --epochs 5 --gpu</code></pre></li>
    <li><strong>Run inference on an image</strong>:<br>
    <pre><code>python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu</code></pre></li>
    <li><strong>Expected Output</strong>:<br>
    <pre><code>Top-5 Predictions:
1. Rose: 0.98
2. Tulip: 0.87
3. Daisy: 0.76
4. Sunflower: 0.65
5. Lily: 0.52</code></pre></li>
</ol>



<p><strong>Example Prediction Output</strong></p>
<p>The model correctly predicted the flower as <strong>Hard-leaved Pocket Orchid</strong> with the highest probability. The top 5 predictions are visualized in the bar chart.</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/5850082e-c132-4488-87d4-0eb05da4707a" alt="Model Prediction Output" width="55%">
</p>
<p align="center">
  <strong><em>Example Prediction Output</em></strong>
</p>

<p><strong>Command Used for Example Prediction:</strong></p>
<pre><code>python predict.py flowers/test/2/image_05100.jpg checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu</code></pre>

<p align="center">
  <img src="https://github.com/user-attachments/assets/be544148-5774-48b5-9c9e-156c51191d69" alt="awsWinter24-project2-badge" width="450">
</p>
<p align="center">
  <strong><em>Project 2 Completion Badge</em></strong>
</p>












<br><br>
<h2 id="project3"><strong>Project 3: Bike Sharing Data EDA and Business Recommendations</strong></h2>

<p> The <strong>Bike Sharing Data EDA</strong> project is for behavioral pattern identification in bike-sharing services to provide actionable insights and business recommendations by analyzing shared bike data from San Fransisco. This project focuses on <strong>exploratory data analysis (EDA)</strong> for pattern recognition. The project explores user behaviors, operational efficiency, and temporal trends to create actionable business insights.</p>

<p><strong>Project Goal:</strong> This project aims to develop practical Python data analysis skills with EDA for bike-sharing pattern analysis. The project follows a custom analysis workflow (implemented in <code>bike_sharing_analysis.ipynb</code>) that includes multiple statistical methods and visualization techniques to perform robust data exploration and insight generation. </p>

<p><strong>List of Objectives</strong></p>
<ul>
    <li>Perform temporal pattern analysis in bike usage and identify key factors influencing demand.</li>
    <li>Analyze user demographics and behavior patterns to segment customer base effectively.</li>
    <li>Evaluate operational efficiency metrics to optimize resource allocation.</li>
    <li>Compare different time periods and locations to determine optimal pricing strategies.</li>
</ul>

<p><strong>Project Structure</strong></p>
<ul>
    <li><code>bike_sharing_analysis.ipynb</code> is the main Jupyter notebook, including data processing, statistical analysis, visualization generation, and insights derivation. This notebook controls the entire analysis workflow.</li>
    <li><code>201902-fordgobike-tripdata.csv</code> is the bike-sharing data as the main resource for analysis and insight generation.</li>
</ul>

<p><strong>Future Improvements:</strong> Future developments might include implementing real-time data analysis capabilities, expanding the analysis to include seasonal pattern recognition, and developing more sophisticated outlier detection algorithms for better accuracy in trip duration analysis.</p>

<p><strong>Prerequisites</strong></p>
<ul>
    <li><strong>Python Environment:</strong> Python 3.x installation required</li>
    <li><strong>Dependencies:</strong> Data analysis frameworks including pandas, matplotlib, seaborn (full list in <code>requirements.txt</code>)</li>
    <li><strong>Dataset:</strong> Valid bike-sharing data files must be present.
</ul>
<p><strong>Execution Guide</strong></p>
<ol type="i">
    <li><strong>Setup</strong>: Install all required packages.<br>
    <pre><code>pip install -r requirements.txt</code></pre></li>
    <li><strong>Launch Jupyter Notebook</strong>: Navigate to the project directory and start Jupyter.<br>
    <pre><code>jupyter notebook bike_sharing_analysis.ipynb</code></pre></li>
    <li><strong>Execute Analysis</strong>: Run all cells sequentially in the notebook to generate analysis and visualizations.</li>
</ol>

<p><strong>Output:</strong> The analysis produces systematic performance metrics, including usage patterns, demographic insights, and operational efficiency measurements. The results include detailed visualizations and actionable business recommendations based on the analyzed patterns.</p>




<br><br>
<h2><strong>Acknowledgements</strong></h2>
Special thanks to <strong>AWS</strong> and <strong>Udacity</strong> for making this scholarship program possible and providing an incredible opportunity to develop skills in AI and machine learning.
