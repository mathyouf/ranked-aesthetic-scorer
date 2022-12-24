# ranked-aesthetic-scorer
Created as a change to the improved-aesthetic-scorer that allows for using ranked/pairwise datasets to train the aesthetic scorer

# Abstract
Currently, it takes in two clip embeddings of shape  (512,). These are put through the  self.aesthetic_score_layers and are loaded using pre-trained weights which output values for individual embeddings in the range of 0-10.

The modification i've made is to add a self.ranking_layers which takes those two scores and then outputs a value to determine which would be rated higher in a human labeled scoring system. The y-value in this case then would be a 0 or 1, depending on if the second or first embedding in the pair is rated higher or not.

Currently, when I train this network, it converges on outputting `0.5` for everything, the loss becomes `0.25`, and the outputs of the self.aesthetic_score_layers blow up to values outside of the 0-10 range.
