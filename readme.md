# Super-Resolution on Computer Texts

This repo examines the super-resolution effects on computer texts with different font sizes and font colors. The main contribution is as follows: 

- Data Collection Tool: A code to automatically make screenshots on Wikipedia.
- Text Image Generator
- Super-Resolution Model: An implementation of TSRN [1] Super Resolution Model and Experiment result on the self-collected dataset


## Explore
We found that TSRN has the problem of generating the color of the texts properly, which increases the difficulty of recognizing the words.

## Possible Solution
- Use an additional Conv Layer specifically for handling the color task
- Design a loss for the color

## Reference

[1] Wang, W., Xie, E., Liu, X., Wang, W., Liang, D., Shen, C., & Bai, X. (2020). Scene text image super-resolution in the wild. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part X 16 (pp. 650-666). Springer International Publishing.
