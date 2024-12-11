<h1><strong>AWS AI/ML Scholarship Program Fall 2024 with Udacity</strong></h1>

<p><strong>Author:</strong> Seungmi Kim (kimsm6397@gmail.com)<br></p>
<p><strong>Repository:</strong> <code>awsAIMLscholarship-fa24<br></code></p>
<p><strong>Last Updated:</strong> December 11, 2024</p>

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
This section will contain the README documentation for the next project: Create Your Own Image Classifier. In this project, a custom image classifier will be developed using TensorFlow or PyTorch, focusing on transfer learning techniques to classify a specific set of images. Additional details will be added as the project progresses.






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
