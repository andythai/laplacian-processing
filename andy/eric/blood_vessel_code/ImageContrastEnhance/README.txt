Human's perception follows Weber's Law. 
According to the law, brighter pixels require more enhancement than darker pixels so that we can perceive the change effectively. 
We effectively apply this fact to design a contrast-enhancement method for images that improves the local image contrast 
by controlling the local image gradient with a single parameter T.

Following are the source code and the executables of this program. To run the code, use the following command.

java  inputimage  outputimage  T  (e.g. java  input.jpg  output.jpg  3.0)


T controls the amount of contrast enhancement achieved. Increasing T increases the amount of contrast enhancement.

Image names must end with .jpg and T must be more than 1. The input image file must be in the same directory with executable codes.