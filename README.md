[cs506 project proposal.md](https://github.com/user-attachments/files/22454930/cs506.project.proposal.md)
		

CS 506 Project Proposal:  
Noise Reduction in Images

By: Ashtosh Bhandari, Zev Fine, Varada Rohokale

Description  
The aim of this project is to develop algorithms and deep learning models that can take noisy, fuzzy, blurred, or low-quality images and transform them into clearer, more understandable versions. This process involves applying deep learning techniques, and reverse engineering gaussian and brownian noise to try to correct the blurring effects. The goal is to restore and preserve the original content as much as possible.

We aim to produce images that, while not completely perfect or perfectly restored, tries to convey enough visual information to identify the mani content/context. This is applicable to broad fields such as surveillance, medical imaging, satellite imagery, and the restoration of historical images.  

Main Goal  
Our main goal is to reduce noise in images, making a clearing image where main ideas that might be blurred out can come through.

Data  
There are two types of data to be collected:

1. Existing image data collections such as existing data set  
   1. [https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)  
   2. [https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset](https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset)  
   3. [https://www.kaggle.com/datasets/gpiosenka/sports-classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification)  
   4. [https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types)  
   5. etc  
2. Photos taken by us  
   

Modeling Data  
To model our data, we will will use:

1. Deep learning  \- Using supervised learning to show what images should be  
2. Mathematical techniques \- 

a. Frequency domain manipulation  
b. Statistics and Probabilistic methods  
c. Filtering   
d. Etc

Data visualization  
To evaluate and present results, we will show

1. Side by side comparison \- an easy way to show how good or bad our methods are is by showing before, after, and target  
2. Over all data \- show the mean, median, mode etc error between our overage targets and show RGB Differences in before and after  
3. Give specific examples \- show where the reduction worked well or poorly (faces, background, building, etc)  
     
   Test Plan

Testing comprises of: 

1. Partition \- Of each data set, set around 75% for training and the rest for testing   
2. Testing \-   
   a. compare each image pixel by pixel its denoised counterpart.   
   b. Use existing image recognition models to see if main ideas are retained 

   i. We will run a clean image and get its classification, then run the denoised counterpart through it and get the similarity.

