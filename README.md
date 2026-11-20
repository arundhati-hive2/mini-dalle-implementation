# Mini DALL-E on Fashion MNIST

A small, from scratch version of the idea behind DALL-E. Give it a caption such as "a sneaker" and a transformer paints a 32 by 32 clothing image to match.

## How it works

1. **Data and captions.** Fashion MNIST images are padded from 28 to 32 pixels so they divide cleanly into an 8 by 8 grid. Each image gets a short caption built from its class, such as "a trouser".
2. **Image tokenizer (discrete VAE).** A small convolutional network turns every image into 64 tokens, an 8 by 8 grid where each cell picks one of 128 codes. The choice is relaxed so it can be trained, and the temperature falls from 1.0 to 0.1 during training so choices harden over time. It has about 224 thousand parameters and trains for 10 epochs.
3. **Transformer.** Caption words and image codes share one vocabulary. A four layer transformer (256 dimensions, 4 heads, about 3.25 million parameters) learns to predict the next token. Image tokens are weighted seven times more than caption tokens in the loss, as in the original DALL-E. Trained for 15 epochs with AdamW.
4. **Generation.** Starting from a caption, the model produces the 64 image tokens one at a time using top 20 sampling, and the tokenizer decodes them into pixels.
5. **Evaluation.** A separate classifier trained on real images checks whether each generated image matches its caption.

## Results

* Classifier accuracy on real images: 0.892
* Classifier accuracy on tokenizer reconstructions: 0.865. This is the practical ceiling for generation, because the transformer can only be as good as the tokenizer's picture of the world.
* Overall caption match on generated images: 0.863

Caption match by class:

* Trouser 1.00, sneaker 0.98, bag 0.97, ankle boot 0.96
* Sandal 0.90, coat 0.89, dress 0.84, t-shirt 0.82
* Pullover 0.71, shirt 0.56

Shirts and pullovers are the weakest classes. They look alike at 32 pixels, even to the classifier.

## CIFAR 10 extension

A second notebook applies the same pipeline to colour images with a larger tokenizer (256 codes, about 898 thousand parameters, 20 epochs) and a longer transformer run (30 epochs). It is an experiment. Colour photographs are much harder for a model this small, so expect rough, blurry output.

## Files

* `mini_dalle_fashion_mnist.ipynb` the main project
* `mini_dalle_cifar10.ipynb` the colour image experiment

## Run it

Open either notebook in Google Colab with a GPU and run the cells in order. The main notebook ends by saving the trained weights to Google Drive. The weights are not stored in this repository.

## Reference

Ramesh et al., Zero Shot Text to Image Generation (2021), the paper behind DALL-E.

Built as a deep learning course project.
