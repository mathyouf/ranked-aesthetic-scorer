# ranked-aesthetic-scorer
A changed version of the [LAION Aesthetics](https://laion.ai/blog//) model called the [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)

## Data

### Toloka
Idea: Use [inter-rater reliability coefficient](https://en.wikipedia.org/wiki/Inter-rater_reliability) as a metric to be directly guessed by the model

### Reddit
Idea: Compare posts which were posted within an hour of each other in the time of day

Idea: Compare subreddits about learning a skill vs pro stuff from that skill (photography, beginnerphotography) (photorealism/ learntodraw)


[Nature Network](https://www.reddit.com/r/sfwpornnetwork/wiki/network/#wiki_nature)

`python data/processURS.py`


# Sites
[Disagreement Images >70%](https://mathyouf.github.io/ranked-aesthetic-scorer/sites/top_10_percent.html)

[HIPLOT of Embeddings](https://mathyouf.github.io/ranked-aesthetic-scorer/sites/aes_hiplot.html)