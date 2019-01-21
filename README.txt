Student data:

Peter Rjabcsenko 1228563
Eugen Havasi 1329709





Entry point of the code:

We used Jupyter Notebook (from the anaconda distribution) for this exercise,
so for the best experience, please use the Jupyter Notebook "src\SimModEx.ipynb" file,
but we also included a pure python script under "src\SimModEx.py" which should also
do the job.





Performance Indicators:

We computed the following performance measures:

precision (micro average): 0.7115665584415585
precision (macro average): 0.5491344342811632
precision (weighted average): 0.6390190567390244
recall (micro average): 0.7115665584415585
recall (macro average): 0.517993875458666
recall (weighted average): 0.7115665584415585
f1 (micro average): 0.7115665584415584
f1 (macro average): 0.4934335949992906
f1 (weighted average): 0.6488477664913387

and the ROC curve, which can be found under "Roc plot.png"

NOTE: we believe that the weighted average measures are the most appropriate
for this experiment, because they take into account the fact that our classes are
unbalanced (i.e. there are many more instances where Kermit doesnt appear as opposed
to those where he does), those again are:

precision (weighted average): 0.6390190567390244
recall (weighted average): 0.7115665584415585
f1 (weighted average): 0.6488477664913387





Timesheets:

Eugen Havasi
16/01/2019 2h brainstorming the big picture
18/01/2019 4h extracting the audio data from the episodes and creating the ground truth files
19/01/2019 6h feature extraction and data preprocessing
20/01/2019 3h choosing a classifier, training/prediction

Peter Rjabcsenko
16/01/2019 2h brainstorming the big picture
18/01/2019 4h installing IDE, libraries, preparing the workspace
19/01/2019 6h feature extraction and data preprocessing
20-21/01/2019 5h evaluating results




Additional Information:

on architecture:
- we used python3 for this exercise, more precisely the anaconda distribution
with Jupyter Notebook (formerly known as iPython), it comes with a lot of
useful preinstalled libraries like "numpy", "sklearn", "matplotlib", etc.
- Additionally we installed the "librosa" audio analysis library, that we used for
feature extraction.
- For training, prediciton and evaluation we used "sklearn".

on our approach:
- We decided to try to do the exercise based only on audio features.
- We used "Adobe Premiere Pro 2018" to extract the audio from the episodes.
- The ground truth was created manually with each second of an episode marked as 1 or 0,
depending whether Kermit could be heard or not. We believe the ground truth
came out rather coarse and that might have influenced our results somewhat.
- We used a power spectrogram as features for our training (a Short-Time Fourier Transform
with the complex part discarded and the real part squared).
For the spectrogram computation we tried different values for Frame Lenght and Hop Length,
but ultimately couldnt see significant a difference for the result so we settles for
Frame Length of 6000 samples and Hop Length 3000. The Sampling Rate that we used is 48000
- Additionally we applied a Mel Scale Filter to our spectrogram with 80 frequency bins as well as
a Log10 Scale so the resulting values were in decibels
- We usd episode 02-01-01 and 02-04-04 for training and 03-04-03 for evaluation, we ended up
not using k-fold cross validation.
- we picked the MPLClassifier (Multi-Layer Perceptron) from the sklearn librariy as our model
with the default parameters

NOTE: A nice overview of all the above things can be seen in "SimModEx.html", which is an html export
of the jupyter notebook that we used (we couldnt make a proper .pdf of it because of some strange bug)