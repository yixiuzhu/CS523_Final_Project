# CS523_Final_Project
1. The data is a text file called “text1.txt” in repo

2. Please also download the original text file for glove vector, which is named as “glove.6B.50d.txt”

3. The only files needed to download are “text1.txt”, “523.py”, and “glove.6B.50d.txt”. Please have them in the same directory

4. To do the testing, after training the model, go to line 292. Change the variable called Random_number manually between 0 and 107532 and run the for loop from line 293 to line 299. Note that the beginning and the end of the “text1.txt” file are not the content of fiction itself, but information such as the publisher. Making the Random_number too small or too large would take the sentences from the beginning or the end, which may cause inaccurate testing results. The number was initially set to be 1000 in script.

*I enabled kera’s checkpoint feature by specifying the file saved as “checkpoint_words”

*The file was originally developed and tested on Windows 10

