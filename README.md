# Moby Eye Tracking

### About

An open source project, aiming to create a robust, accurate, fast eye tracking solution that uses only the existing webcam in your computer or laptop. 

This is still a work in progress, but the current code can be used to efficiently create training data for such a system. To create training data, run cells in the "development" jupyter notebook, up to train_and_preview(), which will launch an interface you can use for training. This shows predictions as grey dots, where you should look at the red dots to train on those points.

### Why?

There are many uses for eye tracking technology, including scientific and commercial research, gaming, and accessibility. Currently, many of these use expensive and bulky external devices. Webcam solutions do exist, but these are very brittle, often requiring the user to sit in an unnaturally still position throughout the experiment.

### Requirements

These are versions used during development, but other recent versions should work. Developed on Windows 10 with jupyter.

    opencv-python 4.2.0.34
    keras 2.3.1
    tensorflow 2.2.0 backend
    face_recognition 1.3.0 (https://pypi.org/project/face-recognition/) 

### Contribute
This is a hobby project of mine, so any help is appreciated. I love pull requests, and you can also leave suggestions/feature requests above. 

Even without python experience you can help out by contributing your face! At the heart of this program is a machine learning system, which I am currently training on lots of pictures of my face. However, like most machine learning, having more data from more people running more devices will help create a more robust, more accurate system. 

Help create training data for this program by running every cell in the jupyter notebook up to train_and_preview(). This will fire up a full-screen interface, where predictions are shown as grey dots, and you continue to look at the red dots as they move. You can watch the accuracy improve as you train, as training happens in real time. Simply make a pull request for your new data and I will incorporate it the project. Please include a .txt file inside your data folder, including of these specifications of your system:
* Laptop model if using a laptop
* Screen resolution
* Screen height and width (mm)
* Webcam make and model if a stand-alone device
* Horizontal webcam position (mm)

### Disclaimer
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.