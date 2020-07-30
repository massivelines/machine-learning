/**
 * P5 has many helper functions that are automatically called
 * setup, mousePressed, draw, map, random
 * below draws a canvas element and when the mouse is pressed
 * it pushes the location into an array and draws a white dot
 */

let x_vals = [];
let y_vals = [];

let m, b;

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function setup() {
  createCanvas(400, 400);

  // tf.variable - Creates a new variable with the provided initial value.
  // tf.scaler - Creates a tf.Tensor with the provided value
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}

function predict(x) {
  const xs = tf.tensor1d(x);
  // y=mx+b;
  // below is same  ys = xs * m + b
  const ys = xs.mul(m).add(b);

  return ys;
}

// pred - y values from predict()
// labels - acutal y values that are part of ys
function loss(pred, labels) {
  return pred.sub(labels).square().mean();
}

function mousePressed() {
  let x = map(mouseX, 0, width, 0, 1);
  let y = map(mouseY, 0, height, 1, 0);

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
    let px = map(x_vals[i], 0, 1, 0, width);
    let py = map(y_vals[i], 0, 1, height, 0);

    point(px, py);
  }

  // Draw line
  const lineX = [0, 1];

  const ys = tf.tidy(() => predict(lineX));
  let lineY = ys.dataSync();
  ys.dispose();
  // ys.print();

  let x1 = map(lineX[0], 0, 1, 0, width);
  let x2 = map(lineX[1], 0, 1, 0, width);

  let y1 = map(lineY[0], 0, 1, height, 0);
  let y2 = map(lineY[1], 0, 1, height, 0);

  strokeWeight(2);
  line(x1, y1, x2, y2);

  // xs.print();
}
