# CS523_Final_Project
1. The data is a text file called “text1.txt” in repo

2. Please also download the original text file for GloVe vector from the shared google drive link(https://drive.google.com/file/d/14CHBn0h96YBxjHHiMvSY4BtKPRVJ_eZC/view?usp=sharing), which is named as “glove.6B.50d.txt”  

3. The only files needed to download are “text1.txt”, “523.py”, and “glove.6B.50d.txt”. Please have them in the same directory

4. To do the testing, after training the model, go to line 292. Change the variable called Random_number manually between 0 and 107532 and run the for loop from line 293 to line 299. Note that the beginning and the end of the “text1.txt” file are not the content of fiction itself, but information such as the publisher. Making the Random_number too small or too large would take the sentences from the beginning or the end, which may cause inaccurate testing results. The number was initially set to be 1000 in script.

*I enabled kera’s checkpoint feature by specifying the file saved as “checkpoint_words”

*The file was originally developed and tested on Windows 10

*If you experienced unexpected error while training the model, it should be caused by Early Stopping or check point from Keras. Please try to distable them in the "callbacks" argument in model.fit in line247 if you do encounter it. The issue results from Keras

