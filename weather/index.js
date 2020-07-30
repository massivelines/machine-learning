const tf = require('@tensorflow/tfjs-node');
const LoadCSVData = require('./data/data');
// const moment = require('moment');

const main = async () => {
  // const jenaWeatherData = new LoadCSVData({
  //   featureColumns: ['T (degC)'],
  //   labelColumns: ['Date Time'],
  //   convertData: {
  //     'Date Time': (data) => {
  //       // const formatDate = moment(data, 'DD-MM-YYYY hh:mm:ss')
  //       // return formatDate.format();
  //       return data;
  //     },
  //   },
  // });

  // const { features, labels } = jenaWeatherData.loadData();

  // console.log('features: ', features);
  // console.log('labels: ', labels);

  const summaryWriter = tf.node.summaryFileWriter('/tmp/tfjs_tb_logdir');

  const text = 'Nostrud nisi et ea laboris do proident do veniam est amet consectetur ullamco consectetur sint.'

  summaryWriter.text('text test', text)

  // for (let step = 0; step < 100; ++step) {
  //   summaryWriter.scalar('dummyValue', Math.sin(2 * Math.PI * step / 8), step);
  // }

  // Constructor a toy multilayer-perceptron regressor for demo purpose.
  // const model = tf.sequential();
  // model.add(
  //   tf.layers.dense({ units: 100, activation: 'relu', inputShape: [200] })
  // );

  // model.add(tf.layers.dense({ units: 1 }));

  // model.compile({
  //   loss: 'meanSquaredError',
  //   optimizer: 'sgd',
  //   metrics: ['MAE'],
  // });

  // // Generate some random fake data for demo purpose.
  // const xs = tf.randomUniform([10000, 200]);
  // const ys = tf.randomUniform([10000, 1]);
  // const valXs = tf.randomUniform([1000, 200]);
  // const valYs = tf.randomUniform([1000, 1]);

  // // Start model training process.
  // await model.fit(xs, ys, {
  //   epochs: 20,
  //   validationData: [valXs, valYs],
  //   // Add the tensorBoard callback here.
  //   callbacks: tf.node.tensorBoard('./tmp/fit_logs_1'),
  // });
};

main();
