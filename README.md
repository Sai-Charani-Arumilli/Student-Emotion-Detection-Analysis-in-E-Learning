# Student-Emotion-Detection-Analysis-in-E-Learning

Emotions of a student during course engagement play a vital role in any learning environment
whether it's in classrooms or in e-learning. Emotion detection methods have been in focus of the
researchers across various disciples to understand the user involvement, effectiveness and
usefulness of the system implemented or to be implemented. Our focus is on understanding and
interpolating the emotional state of the learner during a learning engagement.
Evaluating the emotion of a learner can progressively help in enhancing learning experience and
update the learning contents. In this project, I propose a system that can identify and monitor
emotions of the student in an e-learning environment and provide a real-time feedback mechanism
to enhance the e-learning aids for a better content delivery.
This helps to identify emotions and classify learner involvement and interest in the topic which are
plotted as feedback to the instructor to improve learner experience.
In-order to identify students emotions, a deep learning model inspired from Mini-Xception model is
built and is trained with FER-2013 dataset which contains 35,887 grayscale images where each
image belongs to one of the following classes {“angry”, “disgust”, “fear”, “happy”, “sad”, “surprise”,
“neutral”}. To deal with real-time content, avoiding large number of parameters is the most
important thing. So, I adopted Mini-Xception model which replaces the fully-connected layer that
has more parameters, with global average pooling. Here, I used Haar Cascade classifier as it is
trained with large number of positive and negative images. So that it avoids detecting false positives
better than many other classifiers.
This deep learning model is trained to detect emotions of a person in real-time. At the end, it
analyses the overall emotional behaviour of the person throughout the course and presents reports.
The generated report can be used to analyse and review course content.
