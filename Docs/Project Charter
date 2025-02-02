# Project Charter

## Business Background

Human trafficking is a widespread global problem that affects millions of individuals, often occurring in concealed environments like hotel rooms. Organizations and law enforcement agencies that are dedicated to fighting human trafficking face significant challenges in identifying the locations of victims based on limited visual evidence. Frequently, photographs taken in hotel rooms serve as the only clues to identify these locations. However, the process of manually analyzing these images is time consuming, labor intensive, and prone to errors. With thousands of hotels, each with unique room configurations, accurately pinpointing locations requires an automated, scalable solution.
This project aims to develop a machine learning model capable of matching images of hotel rooms to their respective hotels. By providing data driven, efficient solutions, this project will enable faster response times and improve the success rate of victim rescue operations. With our tools and expertise, the project will address current inefficiencies and set an example for how AI can be used to tackle societal human rights issues.

## Scope

Develop a computer vision model that can efficiently classify hotel images and predict the hotel ID from a new image. Deep machine learning techniques like convolutional Neural Networks (CNNs) or transformer-based models to build a model using previous hotel room pictures to identify a new hotel room. Incorporating an inference pipeline system that is capable of taking an image as its input and outputs the most likely hotel ID. Once the baseline model is sufficient, we will go on to incorporate image augmentation and feature extraction to improve model robustness against variations in lighting, angles, and occlusions. 
Performance metrics will be evaluated using top-k accuracy, precision-recall, and other relevant metrics. Track model performance over iterations and compare new metrics to the baseline to optimize real-world performance. Solution can be incorporated by consumers like law enforcement and NGOs can then upload an image from some source and in turn will get a list of predicted hotel ID’s. This can assist in investigations where these predictions can be used as leads to verify and locate trafficking victims. This model can also potentially be integrated with other law enforcement databases to accelerate investigations, improve resource allocation, and enhance victim recovery efforts.

## Personnel

* Sneha Gonipati
	* Skills: Data preprocessing in Pytorch/Pandas/R, Data Analysis and Presentation in Python, Data Exportation to Various Sources, Model Creation with NumPy and Sk.Learn, Basic Python UI Construction 
  * Roles: Model Engineer
* Zinnia Waheed
  * Skills: Data preprocessing/manipulation in Python, machine learning, database/SQL, numpy, beautifulsoup, exploratory data analysis 
  * Roles: Data Engineer, Model Engineer 
* Aiah Aly
  * Skills: Graph/image based classification, (GNNs) 
  * Roles: Model Engineer
* Heron Ziegel
  * Skills: Front end, UI, graphics, back end, database/SQL, writing, data manipulation, GitHub, basic machine learning 
  * Roles: GitHub coordinator, front end if needed, ML programming
* Faisal Shaikh
  * Skills: Data Preprocessing and Management (data wrangling, cleaning, data augmentation/manipulation) in Python/R, Exploratory Data Analysis in Python/R, basic SQL/database
  * Roles: Data engineer (preprocessing/EDA)

## Metrics

### How will we quantify the success of the project?
The performance will be evaluated based on the Mean Average Precision at 5 (MAP@5) metric, which calculates the precision of the top five predictions made by the model for every image. MAP@5 assigns higher scores to models that rank the correct label earlier in their predictions. This metric specifically aligns the model's effectiveness with the project objective of effectively detecting hotels and hence is highly pertinent to practical applications such as facilitating law enforcement efforts in the fight against human trafficking.

### What are the common metrics used by others working in the domain?
On the image recognition and ranking tasks, a number of measures are most commonly utilized, such as Mean Average Precision (MAP), precision and recall, top-k accuracy, and F1-score. MAP calculates precision for all the predictions and is most suitable to applications where ranking and relevance are the focus. Precision calculates the proportion of relevant results out of the total predictions, whereas recall calculates the proportion of correctly predicted relevant results out of the overall set of possible relevant results. Top-k accuracy checks whether the true label is among the top k predictions, and Top-5 accuracy is one of the standard evaluations. F1-score as the harmonic mean of precision and recall is utilized to balance between the two to yield a balanced metric for classification tasks. In this project, MAP@5 is utilized with an emphasis on the top five predictions to guarantee an accurate and pertinent ranking to render it specifically suitable for improving hotel matches here.

### What is a quantifiable metric?
The MAP@5 score is the main measurable metric that assesses the model's capacity to accurately rank the correct hotel label in its top five predictions for every image. By yielding a definite numerical score, the MAP@5 score makes it simple to compare various models and enables improvement to be tracked over time.

### How will we measure the metric?
The MAP@5 measure will be evaluated by comparing the model's predictions on a test or validation set of images where we know the ground-truth labels. The MAP@5 score overall will be the average precision over the set of all images. Further, the model's performance will also be compared against a baseline, e.g., a naive or an already-existing model, to see if there is an improvement. The competition leaderboard establishes benchmarks, enabling direct comparison of the MAP@5 score against others and thereby guaranteeing an intensive and systematic assessment process.

## Plan

* Deliverable Timeline
  * Phase 1 Project Demo due 2/17
    * Data exploration (visualization, correlation coefficients, outliers, sparsity, etc.)
    * The most simplistic model deliverable 
  * Bimonthly Project Report 1 due 2/19
  * Project Charter Revised due 3/12
  * Phase 2 Project Demo due 3/25
    * Final model with no masks 
  * Bimonthly Project Report 2 due 3/31
  * Data Report due 4/7
  * Model + Performance Report due 4/14
 * Phase 3 Project Demo + Interactive Dashboard Demo due 4/28
  * Model with masking interactive dashboard 

## Architecture

### What data do we expect?
The data was collected by taking pictures of various hotel rooms, making sure that they were relatively natural and captured most if not all of the room. There are a different number of pictures for each hotel room. For each hotel, there is a subfolder present on the dataset with a unique numerical identifier, and the hotel rooms consist of places from across the globe. Each picture is represented as a JPG file. 
The occlusions are represented as red blocks covering random portions of the picture, with transparent portions for the rest of the image. These masks are represented as PNGs, and are used to simulate elements such as people, jackets, or other large objects that might cover portions of the image. 
This data was not collected by personnel but instead downloaded from the “Hotel-ID to Combat Human Trafficking 2022 - FGVC9” competition from Kaggle. If the data from this project is insufficient, the group may pull from the previous year's competition with similar data in order to fine-tune the model.

### What do we expect and how will it be stored/operated on?
The on-prem will likely be Juypter notebooks on the local computers of the personnel of the project. The model will be built utilizing Pytorch, and the version history protection and data storage will be done using GitHub. Any advanced packages will be found using the advanced machine learning cluster provided by Temple University. 
We expect the machine learning model to be a similarity model, which will receive the input data and find the pictures from the database that most closely match the initial input. 

### On what platform would the client/user access any deliverable services (e.g. dashboard) from the project?
As of right now, the dashboard for delivering the services has not been finalized, but the dashboard will likely include a source to upload the photo in question. Once the photograph is uploaded, a list of the hotels with the most likely hotel match will be presented to the user. Currently, the hotel names are deidentified, but if a dataset with photographs labeled with the hotels in question is given, then along with the list of hotel names and addresses will be given so that the officials using the program will have all the necessary information immediately. 

## Communication
### Trello
We have created a Trello board for the project, which can be continuously updated. There are three sections: “To Do,” “Doing,” and “Done.” As we determine specific deliverables we will add them to the “To Do” section. When someone starts working on a deliverable they can add their name to the card and move it to “Doing,” then finally to “Done” when it’s completed. We may want to update this with more lists such as “Needs Review” and “Deployed” depending on the needs of the project.

### Group Chat
We have created a text message based group chat with all members of the team. We will use this to send messages to the group, plan meetings and stay on top of deliverables.

### GitHub
We will use GitHub for version control to streamline this project. We will create new branches for different elements and send pull requests to the main branch once an element is ready to be included in the project.

### Meetings
We will schedule times to meet outside of class periodically during this project. Meetings could be in-person, particularly after the lab session on Tuesdays. Or they could be virtual using either Discord or Zoom.






