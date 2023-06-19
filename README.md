# analyze_sections
Identify the various sections (bridge, chorus, etc) of a song.

---

## Background

A lot of recent AI research has been focused on data rather than the actual mechanisms of machine learning. For creating an AI DJ, my first instinct was to find, or create, a dataset with a bunch of strong DJ transitions; perhaps I could download a bunch of [Big Bootie Mixes](https://soundcloud.com/two-friends/sets/big-bootie-mixes) off of SoundCloud. But after some deeper thought, I realized the following: DJing is less

> How do I mix between songs?

and more

> Where do I transition into the next track?

Most DJs use the same few mixing processes to transition between songs, all of which can be easily approached programmatically. The greater struggle is finding where to mix two songs. I will use machine learning to identify the different sections of a song, labelling the chorus, verse, buildup, outro -- the list goes on. I need to label the sections of a bunch of songs in my personal music library that I will use as training data. This progress will be analogous to setting Hot Cues for anyone interested in the DJing terminology.


## Data

My personal song library contains >2000 songs, and I will use this library as training data for my neural networks. I will exclude songs labelled under the `Classical`, `Jazz`, `Folk`, `World`, and `Soundtrack` genres for a variety reasons: they tend to have variable key and tempo and also have unusual forms rarely seen in DJing. I will label a dataset that, for each song, includes the song section type and the time at which each section starts (measured in seconds to two decimal places). For example, for song A by artist B, the data will look like this (with some columns like key and BPM, which are constant throughout a song, omitted from the middle):

```
SONG    ARTIST    ...     SECTION     START_TIME
A       B         ...     intro         0.00
A       B         ...     verse        12.42
A       B         ...     chorus       48.34
A       B         ...     verse        96.58
A       B         ...     chorus      135.77
A       B         ...     outro       170.01
```


## Machine Learning

I plan to use a sliding window Convolutional Neural Network (CNN) to analyze patterns in the audio data, such as the introduction of new instruments or chords that often denote the beginning of a new section in the music. However, as I am for determining [tempo](https://github.com/pnlong/determine_tempo) and [key](https://github.com/pnlong/determine_key), I am struggling to decide whether to use a CNN or Long Short Term Memory -- perhaps I can [combine the two](https://www.mathworks.com/help/deeplearning/ug/sequence-classification-using-cnn-lstm-network.html)?


## Output

I will train three neural networks, which will output the following values:

- **Change in Section**: Treating this as a binary classification problem, a sliding window CNN will be used on a song. This CNN will output the probability that the current window contains a shift from one song section to the next. An output value >=0.5 suggests that the window contains a shift to a new section, while a value <0.5 suggests the window does not contain a shift. I imagine that if graphed, section changes would be made apparent by elevated regions, with the peak of each region being the most likely location of the section change.
- **Section Timestamps**: Now treating this as a linear regression problem, for every window suspected of having a section change as calculated by the Change in Section neural network, this CNN will output the predicted location (in seconds into the audio sample) of the section change. I will then use this data, as well as the location of the window in the context of the full song, to generate a list of timestamps at which new sections begin.
- **Section Labels**: Treating this as a multiclass classification problem, for every section identified by the Section Timestamps neural network, I will label the section as one of five values: `intro`, `outro`, `verse`, `chorus`, or `buildup`. I will probably using a sliding window CNN whose final layer is a SoftMax activation function that yields a vector of five values; the window will take on the label value in the output vector with the greatest probability. The most common label value across the multiple windows in a section determines the section label. I can now label the various sections of a song. If adjacent sections have the same labels, this is probably the result of a mistake in the Section Timestamps neural network, and I will combine these sections by removing the middle timestamp(s).


---

## Software

### *.py*
