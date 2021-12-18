# Word Spotting in Handwritten Documents

Because of the huge variability and noise in historical handwritten documents it is inacurate to use OCR techniques to extract information. This project, in cooperation with TU Dortmund, focuses on word spotting techniques with Deep learning in order to digitize information in handwritten documents and accelerate historic research.

Training of Deep neural networks for a two stages process: First, words or hieroglyphics are detected with deep neural networks, and second, features are extracted in an end-to-end fashion with a custom loss function that allows us to encode both the words characters and the position.
