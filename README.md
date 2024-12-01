# MSM_FinalProject
Final Project of Mathematical and Statistical Methods. Project focus on PCA signal compression.

## Abstract 
Principal Component Analysis (PCA) is a powerful technique that enables us to retain maximum information using a minimal set of variables. Its applications are diverse, and one notable example is signal compression. In physics, the signals we have to deal with are delivered by the detectors. Here, to asses more easily the method, we will work with sound signal. We developed a Matlab script using essential functions to treat with sound data, and the equations used are derived from lectures. We analyzed multiple sound signals that allow us to obtain conclusions related to the nature of distortion and the behavior of a compressed signal. Additionally we implemented the use of an AI to compare our results. This text outlines our methodology, presents results, engages in discussions, and draws conclusions based on our findings.

## Conclusions
In this project we build a PCA encoder and its decoder, we use it to compress different audio signals obtaining satisfactory results.
- We analysed our original data and we conclude on the importance of standardization process for a correct PCA compression.
- We used Deepting to compare our results related to the number of principal components requerid to understand an speech. We conclude that, for a grouping of 100, we need at least 30 components to understand the message, while the mentioned AI need at least 70. We considered that this is due to the complexity of the communication process.
- We calculated the distortion values for different combinations of space saved and grouping, here we conclude that, for signals with little noise, we can consider that the distortion is a constant value for a fixed saved space, and it has a linear relation with the saved space.
- We calculate and compare distortion values using various formulas, discovering that despite some differences, we can statistically affirm that both approaches are equivalent. Therefore, both formulas are deemed valid for computing distortion.
- We learn how to compress signals, which reduces the amount of data required to represent a signal. This is particularly crucial in experimental physics, where large data sets are often generated. Efficient data storage allows researchers to store and manage experimental results more effectively
