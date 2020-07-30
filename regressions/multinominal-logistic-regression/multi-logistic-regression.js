/**
 * Multiple Classification Options
 * Copy of logistic-regression.js
 * Notes
 * Mean square error
 * b = y intercept on a xy graph (0, b), approximate line that runs through all points
 * where it hits the y axis
 * m = slope of the line
 */

const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.costHistory = [];
    // this.bHistory = [];

    // default options
    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
        batchSize: 10,
        decisionBoundary: 0.5,
      },
      options
    );

    // (m, b)
    // generate tensor with zeros (dynamic columns)
    // UPDATED: changed so weights has dynamic shape
    // this.weights = tf.zeros([this.features.shape[1], 1]);
    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
  }

  gradientDescent(features, labels) {
    // Mean square error (MSE)
    // sigmoid converts to logistic
    // UPDATE: softmax - it changes return to percentages [[0.2, 0.3, 0.5]]
    // - const currentGuesses = features.matMul(this.weights).sigmoid();
    const currentGuesses = features.matMul(this.weights).softmax();
    const differences = currentGuesses.sub(labels);

    // (bSlope, mSlope)
    // (transpose, have to change direction of array so that it can be multiplied)
    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    // (m, b)
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    // number of batches to run gradientDescent with
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    );

    // for (let i = 0; i < this.options.iterations; i++) {
    //   // console.log(this.options.learningRate)
    //   // this.bHistory.push(this.weights.get(0, 0));
    //   this.gradientDescent();
    //   this.recordMSE();
    //   this.updateLearningRate();
    // }

    // setup for batch
    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const { batchSize } = this.options;
        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1]
        );

        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

        this.gradientDescent(featureSlice, labelSlice);
      }

      this.recordCost();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return (
      this.processFeatures(observations)
        .matMul(this.weights)
        // UPDATE: softmax - it changes return to percentages [[0.2, 0.3, 0.5]]
        // - .sigmoid()
        .softmax()
        // UPDATE: add argMax
        // - .greater(this.options.decisionBoundary)
        // - .cast('float32')
        // argMax returns the index with the largest value across axes of a tensor.
        .argMax(1)
    );
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    // UPDATE: add argMax
    testLabels = tf.tensor(testLabels).argMax(1);

    // UPDATE: add notEqual remove abs
    // (compares predictions [] and labels [], returns [] with 0 where equal and 1 not equal,)
    // const incorrect = predictions.sub(testLabels).abs().sum().get();
    const incorrect = predictions.notEqual(testLabels).sum().get();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  processFeatures(features) {
    // (if use mean and variance on features, must also use on testFeatures)
    features = tf.tensor(features);

    // if mean and variance are defined, calculate, else do first time standardize()
    // (could be moved to standardize)
    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    // generate column of 1,s, merge with features
    // (must be after standardize, else effect column of 1's, (javaScript rounding errors))
    // (this.features.shape[0] is length of data)
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    // undefined on build,
    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }

  // Records MSE to array, newest first
  recordCost() {
    // UPDATE: softmax - it changes return to percentages [[0.2, 0.3, 0.5]]
    // - const guesses = this.features.matMul(this.weights).sigmoid();
    const guesses = this.features.matMul(this.weights).sigmoid();

    const termOne = this.labels.transpose().matMul(guesses.log());
    const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(guesses.mul(-1).add(1).log());

    const cost = termOne
      .add(termTwo)
      .div(this.features.shape[0])
      .mul(-1)
      .get(0, 0);

    this.costHistory.unshift(cost);
  }

  // Self adjusting leaning rate
  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }

    // if mseHistory is increasing (more incorrect), reduce leaning rate
    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;
