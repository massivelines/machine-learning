/**
 * P5 has many helper functions that are automatically called
 * setup, mousePressed, draw, map, random
 * below draws a canvas element and when the mouse is pressed
 * it pushes the location into an array and draws a white dot
 */

let x_vals = [];
let y_vals = [];

let a, b, c, d;

const learningRate = 0.1;
const optimizer = tf.train.adam(learningRate);

function setup() {
  createCanvas(400, 400);

  // tf.variable - Creates a new variable with the provided initial value.
  // tf.scaler - Creates a tf.Tensor with the provided value
  a = tf.variable(tf.scalar(random(-1, 1)));
  b = tf.variable(tf.scalar(random(-1, 1)));
  c = tf.variable(tf.scalar(random(-1, 1)));
  d = tf.variable(tf.scalar(random(-1, 1)));
}

function predict(x) {
  const xs = tf.tensor1d(x);
  // y = ax^3 + bx + c
  const ys = xs
    .pow(tf.scalar(3))
    .mul(a)
    .add(xs.square().mul(b))
    .add(xs.mul(c))
    .add(d);
  return ys;
}

// pred - y values from predict()
// labels - acutal y values that are part of ys
function loss(pred, labels) {
  return pred.sub(labels).square().mean();
}

function mousePressed() {
  let x = map(mouseX, 0, width, -1, 1);
  let y = map(mouseY, 0, height, 1, -1);

  x_vals.push(x);
  y_vals.push(y);
}

function draw() {
  tf.tidy(() => {
    // limit running to only when data is in arrays
    if (x_vals.length > 0) {
      const ys = tf.tensor1d(y_vals);

      optimizer.minimize(() => {
        return loss(predict(x_vals), ys);
      });
    }
  });

  background(0);

  stroke(255);
  strokeWeight(8);

  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], -1, 1, 0, width);
    let py = map(y_vals[i], -1, 1, height, 0);

    point(px, py);
  }

  // Draw curve
  const curveX = [];
  for (let x = -1; x < 1.01; x += 0.05) {
    curveX.push(x);
  }

  const ys = tf.tidy(() => predict(curveX));
  let curveY = ys.dataSync();
  ys.dispose();

  beginShape();
  noFill();
  stroke(255);
  strokeWeight(2);

  for (let i = 0; i < curveX.length; i++) {
    let x = map(curveX[i], -1, 1, 0, width);
    let y = map(curveY[i], -1, 1, height, 0);
    vertex(x, y);
  }

  endShape();
}
